import json
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

D = Path(__file__).parent / "data"
MODELS = ["GBLUP", "LightGBM_all", "LightGBM_top200", "Ridge_top200",
          "PCA_top200", "Loci_Ridge", "Loci_LightGBM"]

st.set_page_config(page_title="Rice529 Marker Explorer", layout="wide")


@st.cache_data
def load():
    r2 = pd.read_csv(D / "model_r2.csv")
    loci = pd.read_csv(D / "our_loci_vs_zhou.csv")
    zs = pd.read_csv(D / "zhou2026_sites.csv")
    zr = pd.read_csv(D / "zhou2026_r2.csv")
    perm = json.loads((D / "overlap_test.json").read_text())
    return r2, loci, zs, zr, perm


r2, loci, zs, zr, perm = load()


def nice(t):
    return t.replace("_", " ")


st.title("Rice529 Marker Explorer")
st.caption("RiceVarMap v2.0 Rice529: 529 accessions, 5,000 SNPs, 10 traits. "
           "All R2 values are nested 5-fold cross-validation.")

t1, t2, t3, t4 = st.tabs(["Overview", "Trait explorer", "All lead markers",
                          "Compare with Zhou 2026"])

# Overview
with t1:
    c = st.columns(4)
    c[0].metric("Best mean R2", "0.424", "LightGBM, all markers", delta_color="off")
    c[1].metric("Mean leakage inflation", "+0.141", "R2 added by naive selection", delta_color="off")
    c[2].metric("Candidate loci", str(len(loci)), f"{loci.lead_marker.nunique()} distinct lead markers", delta_color="off")
    c[3].metric("Class accuracy", "0.576", "chance 0.333", delta_color="off")

    mean = r2[MODELS].mean().reset_index()
    mean.columns = ["model", "mean_r2"]
    ch = alt.Chart(mean).mark_bar(color="#a3191f").encode(
        x=alt.X("mean_r2:Q", title="Mean nested R2", scale=alt.Scale(domain=[0, 0.5])),
        y=alt.Y("model:N", sort="-x", title=None),
        tooltip=["model", alt.Tooltip("mean_r2:Q", format=".3f")])
    st.subheader("Mean R2 over ten traits")
    st.altair_chart(ch, width="stretch")

    lk = r2[["trait", "naive_r2_top200", "LightGBM_top200"]].copy()
    lk["trait"] = lk.trait.map(nice)
    lk = lk.melt("trait", var_name="selection", value_name="r2")
    lk["selection"] = lk.selection.map({"naive_r2_top200": "Naive (leaky)",
                                        "LightGBM_top200": "Nested (honest)"})
    ch2 = alt.Chart(lk).mark_bar().encode(
        y=alt.Y("trait:N", title=None),
        x=alt.X("r2:Q", title="R2, top-200 LightGBM"),
        yOffset="selection:N",
        color=alt.Color("selection:N", scale=alt.Scale(range=["#a2a2a7", "#a3191f"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["trait", "selection", alt.Tooltip("r2:Q", format=".3f")])
    st.subheader("Leakage check: naive vs nested marker selection")
    st.altair_chart(ch2, width="stretch")

# Trait explorer
with t2:
    tr = st.selectbox("Trait", r2.trait, format_func=nice)
    row = r2[r2.trait == tr].iloc[0]
    sub = loci[loci.trait == tr].sort_values("p_value")
    best = row[MODELS].astype(float)
    c = st.columns(4)
    c[0].metric("Best R2", f"{best.max():.3f}", best.idxmax(), delta_color="off")
    c[1].metric("Leakage inflation", f"+{row.naive_r2_top200 - row.LightGBM_top200:.3f}")
    c[2].metric("Candidate loci", str(len(sub)))
    c[3].metric("Class accuracy", f"{row.class_acc:.3f}")

    left, right = st.columns(2)
    with left:
        m = best.reset_index()
        m.columns = ["model", "r2"]
        st.altair_chart(alt.Chart(m).mark_bar(color="#a3191f").encode(
            x=alt.X("r2:Q", title="Nested R2"), y=alt.Y("model:N", sort="-x", title=None),
            tooltip=["model", alt.Tooltip("r2:Q", format=".3f")]), width="stretch")
    with right:
        sub2 = sub.assign(mb=sub.position / 1e6, logp=-sub.p_value.apply(lambda v: __import__("math").log10(v)))
        st.altair_chart(alt.Chart(sub2).mark_circle(size=80, color="#a3191f").encode(
            x=alt.X("chr:O", title="Chromosome"),
            y=alt.Y("logp:Q", title="-log10 p"),
            tooltip=["lead_marker", "chr", "position", alt.Tooltip("p_value:Q", format=".1e"), "partial_r"]),
            width="stretch")
    st.dataframe(sub[["lead_marker", "chr", "position", "p_value", "partial_r", "n_tagged",
                      "nearest_zhou_site", "distance_kb"]],
                 hide_index=True, width="stretch")

# All lead markers
with t3:
    st.write(f"{len(loci)} loci across 10 traits, {loci.lead_marker.nunique()} distinct lead markers. "
             "A lead marker is the strongest marker in a 250 kb window.")
    c = st.columns(3)
    trs = c[0].multiselect("Traits", sorted(loci.trait.unique()), format_func=nice)
    chs = c[1].multiselect("Chromosomes", list(range(1, 13)))
    pmax = c[2].select_slider("p-value at most", [1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-8], value=1e-2)
    f = loci.copy()
    if trs:
        f = f[f.trait.isin(trs)]
    if chs:
        f = f[f.chr.isin(chs)]
    f = f[f.p_value <= pmax]
    shared = loci.groupby("lead_marker").trait.nunique()
    f["n_traits"] = f.lead_marker.map(shared)
    st.dataframe(f.sort_values(["chr", "position"]), hide_index=True, width="stretch")
    st.download_button("Download as CSV", f.to_csv(index=False), "lead_markers.csv", "text/csv")
    st.subheader("Lead markers shared by two or more traits")
    sh = (loci.groupby(["lead_marker", "chr", "position"]).trait
          .apply(lambda s: ", ".join(sorted(nice(x) for x in s))).reset_index())
    sh["n_traits"] = sh.trait.str.count(",") + 1
    st.dataframe(sh[sh.n_traits >= 2].sort_values("n_traits", ascending=False),
                 hide_index=True, width="stretch")

# Zhou 2026
with t4:
    st.write("Zhou et al. 2026, Frontiers in Plant Science 17:1942478, predicted traits from leaf "
             "metabolites in 524 accessions and ran GWAS on the predicted values. Below, their "
             "reported candidate genes and SNPs are compared with our lead markers by position.")
    st.subheader("Prediction accuracy (best model per trait)")
    cmp = zr.merge(r2[["trait"] + MODELS], on="trait")
    cmp["our_best_r2"] = cmp[MODELS].max(axis=1)
    cmp["trait"] = cmp.trait.map(nice)
    st.dataframe(cmp[["trait", "zhou_best_r2", "zhou_model", "our_best_r2"]],
                 hide_index=True, width="stretch")
    st.caption("The tasks differ (metabolites vs SNPs, different panels), so this is not a ranking.")

    st.subheader("Our lead markers within 250 kb of a Zhou 2026 site")
    near = loci[loci.within_250kb].sort_values(["chr", "position"])
    st.dataframe(near[["trait", "lead_marker", "chr", "position", "p_value", "nearest_zhou_site",
                       "zhou_traits", "distance_kb", "same_trait"]],
                 hide_index=True, width="stretch")
    c = st.columns(3)
    c[0].metric("Distinct lead markers near a Zhou site",
                f"{perm['obs_distinct']} of {perm['n_distinct']}",
                f"random expectation {perm['exp_distinct']:.1f}, p = {perm['p_distinct']:.2f}",
                delta_color="off")
    c[1].metric("Trait-locus pairs near a Zhou site", f"{perm['obs_any']} of {len(loci)}",
                f"random expectation {perm['exp_any']:.1f}, p = {perm['p_any']:.3f}", delta_color="off")
    c[2].metric("Pairs with the same trait as Zhou", str(perm["obs_same"]),
                f"random expectation {perm['exp_same']:.1f}, p = {perm['p_same']:.2f}", delta_color="off")
    st.info("Overlap with Zhou 2026 is not clearly above chance when each marker is counted once "
            "(p = %.2f). Treat matches as leads to check, not as confirmation." % perm["p_distinct"])
    st.caption(perm["note"])

    st.subheader("Positions on the 12 chromosomes")
    a = loci.assign(source="Our lead marker", label=loci.lead_marker)[["chr", "position", "source", "label"]]
    b = zs.assign(source="Zhou 2026 site", label=zs.zhou_site)[["chr", "position", "source", "label"]]
    pts = pd.concat([a, b])
    pts["mb"] = pts.position / 1e6
    st.altair_chart(alt.Chart(pts).mark_tick(thickness=2, size=18).encode(
        x=alt.X("mb:Q", title="Position (Mb)"),
        y=alt.Y("chr:O", title="Chromosome"),
        color=alt.Color("source:N", scale=alt.Scale(range=["#a2a2a7", "#a3191f"]),
                        legend=alt.Legend(orient="bottom", title=None)),
        tooltip=["source", "label", "chr", "position"]), width="stretch")
    st.subheader("Zhou 2026 sites used")
    st.dataframe(zs, hide_index=True, width="stretch")
