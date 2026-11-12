# Rice529 Marker Explorer (Streamlit)

Run on your laptop:

    pip install -r requirements.txt
    streamlit run app.py

It opens in the browser at http://localhost:8501.

Tabs:
- Overview: mean R2 per model, leakage check.
- Trait explorer: pick a trait, see model R2, its loci and a chromosome plot.
- All lead markers: all 247 loci (185 distinct lead markers), filters, CSV download.
- Compare with Zhou 2026: R2 side by side, our lead markers within 250 kb of their reported genes/SNPs, permutation test.

Data files (in data/):
- our_loci.csv: all loci printed by the notebook (Section 8).
- our_loci_vs_zhou.csv: same, plus nearest Zhou 2026 site and distance.
- zhou2026_sites.csv: 7 candidate genes and 12 SNPs taken from the Zhou 2026 paper text.
- model_r2.csv, zhou2026_r2.csv, overlap_test.json.

Notes:
- Zhou 2026 positions are read from variant IDs in their text. Their full supplementary tables were not available.
- The app shows saved results. It does not retrain models or predict new plants.
