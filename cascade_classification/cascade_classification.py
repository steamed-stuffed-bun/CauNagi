"""
Phase 1: Multi-dimensional Quantitative Gene Classification
==========================================================

Answers Q1: What is VEXAS's "transcriptional cascade"?

Steps:
  1. Load 4 cell-type driver gene files + overall
  2. [0,1] global min-max normalization + per-celltype z-score
  3. CTS (Cell Type Specificity Index) via Shannon entropy
  4. HPS (Hierarchical Propagation Score) weighted sum
  5. Layer Consistency (1 - CV)
  6. 5-category gene classification
  7. Propagation depth + 10,000 permutation test
  8. Mechanism & evidence annotation
  9. Save gene_cascade_classification.csv + stats JSON
"""
import sys, os, json
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
BASE_DIR = os.path.join(PROJECT_ROOT, "results", "driver_genes")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
COL_ORDER = ["HSPC", "GMP", "Monocyte", "Neutrophil"]
HPS_W = np.array([0.30, 0.25, 0.25, 0.20])
MIN_RAW = 0.10
N_PERM = 10000
LOG2_4 = np.log2(4)

# ================================ Load ================================
def load_data():
    ct_scores = {}
    for ct in COL_ORDER:
        df = pd.read_csv(os.path.join(BASE_DIR, f"{ct}_driver_genes.csv"))
        df["gene"] = df["gene"].str.upper().str.strip()
        ct_scores[ct] = df.groupby("gene")["driver_score"].max().to_dict()
        print(f"  {ct}: {len(df)} genes")

    all_genes = sorted(set().union(*[ct_scores[c].keys() for c in COL_ORDER]))
    print(f"  Union: {len(all_genes)} genes")

    raw = pd.DataFrame(0.0, index=all_genes, columns=COL_ORDER)
    for ct in COL_ORDER:
        for g, s in ct_scores[ct].items():
            raw.at[g, ct] = max(0.0, float(s))

    ov_df = pd.read_csv(os.path.join(BASE_DIR, "overall_driver_genes.csv"))
    ov_df["gene"] = ov_df["gene"].str.upper().str.strip()
    ov = ov_df.set_index("gene")[["driver_score", "gene_category"]].to_dict("index")
    ov_rank = {g: i+1 for i, g in enumerate(ov_df["gene"])}

    meta = pd.DataFrame({"overall_score": np.nan, "overall_rank": np.nan,
                         "overall_category": "other"}, index=all_genes)
    for g in all_genes:
        if g in ov:
            meta.at[g, "overall_score"] = ov[g]["driver_score"]
            meta.at[g, "overall_category"] = ov[g].get("gene_category", "other")
        if g in ov_rank:
            meta.at[g, "overall_rank"] = ov_rank[g]
    print(f"  Overall: {meta['overall_rank'].notna().sum()} genes")
    return raw, meta

# ========================= Normalize ===========================
def normalize(raw):
    mn, mx = float(raw.values.min()), float(raw.values.max())
    print(f"  Global MIN={mn:.6f}, MAX={mx:.6f}")
    S = ((raw - mn) / (mx - mn)).clip(0, 1) if mx > mn else pd.DataFrame(0.5, index=raw.index, columns=COL_ORDER)
    Z = pd.DataFrame(0.0, index=raw.index, columns=COL_ORDER)
    for ct in COL_ORDER:
        mu, sd = float(raw[ct].mean()), float(raw[ct].std(ddof=1))
        print(f"  {ct}: mu={mu:.6f}, sigma={sd:.6f}")
        if sd > 1e-12:
            Z[ct] = (raw[ct] - mu) / sd
    return S, Z, mx

# ========================= Metrics ==============================
def compute_cts(S):
    eps = 1e-12
    rs = S[COL_ORDER].sum(axis=1)
    P = S[COL_ORDER].div(rs, axis=0).fillna(0)
    H = -(P.values * np.log2(P.values + eps)).sum(axis=1)
    cts = 1 - H / LOG2_4
    cts = pd.Series(cts, index=S.index, name="CTS")
    cts[rs < eps] = 0
    return cts

def compute_hps(S):
    return pd.Series(S[["HSPC","GMP","Neutrophil","Monocyte"]].values @ HPS_W, index=S.index, name="HPS")

def compute_lc(S):
    v = S[["HSPC","GMP","Neutrophil","Monocyte"]].values
    mu = v.mean(axis=1); std = v.std(axis=1, ddof=0)
    lc = np.where(mu > 0, 1 - std / mu, 0)
    return pd.Series(lc, index=S.index, name="layer_consistency")

# ======================== Classify ==============================
def classify(S, lc, min_s):
    """
    Calibrated thresholds based on actual data distribution (min_S at rank 35=0.55).
    Priority-ordered.
    """
    genes = S.index.tolist()
    a = S["HSPC"].values; b = S["GMP"].values
    c = S["Monocyte"].values; d = S["Neutrophil"].values
    L = lc.values
    cats = []
    for i in range(len(genes)):
        sh, sg, sm, sn = a[i], b[i], c[i], d[i]
        lv = L[i]
        # Cascade_Core_Uniform: min_S >= 0.45 AND consistent across layers
        if sh > 0.45 and sg > 0.45 and sm > 0.45 and sn > 0.45 and lv >= 0.65:
            cats.append("Cascade_Core_Uniform")
        # Cascade_Core_Decay: present in all 4, not uniform, HSC dominates (ratio relaxed to 1.3)
        elif sh > min_s and sg > min_s and sm > min_s and sn > min_s and lv < 0.65 and sh > sn * 1.3:
            cats.append("Cascade_Core_Decay")
        # HSC_Initiator: HSC-dominant, others minimal
        elif sh > 0.35 and sg < 0.20 and sn < 0.20 and sm < 0.20:
            cats.append("HSC_Initiator")
        # GMP_Propagator: GMP-dominant with downstream signal
        elif sg > 0.30 and (sn > 0.20 or sm > 0.20) and sh < 0.30:
            cats.append("GMP_Propagator")
        # Mono_Amplifier: Monocyte-specific amplifier (lower S threshold to 0.25)
        elif sm > 0.25 and max(sh, sg, sn) < 0.15:
            cats.append("Mono_Amplifier")
        # Neut_Amplifier: Neutrophil-specific amplifier (lower S threshold to 0.25)
        elif sn > 0.25 and max(sh, sg, sm) < 0.15:
            cats.append("Neut_Amplifier")
        else:
            cats.append("Other")
    r = pd.DataFrame({"gene": genes, "cascade_category": cats})
    return r

# ====================== Permutation ============================
def perm_test(raw, min_r=MIN_RAW, n=N_PERM):
    ag = raw.index.values
    ln = {}
    for ct in COL_ORDER:
        ln[ct] = (raw[ct] > min_r).sum()
        print(f"  {ct}: {ln[ct]} genes > {min_r}")
    obs = len(set(raw.index[raw["HSPC"] > min_r]) & set(raw.index[raw["GMP"] > min_r]) &
              set(raw.index[raw["Monocyte"] > min_r]) & set(raw.index[raw["Neutrophil"] > min_r]))
    rng = np.random.RandomState(42)
    nd = np.zeros(n, dtype=int)
    for i in range(n):
        sets = [set(rng.choice(ag, size=ln[ct], replace=False)) for ct in COL_ORDER]
        nd[i] = len(sets[0] & sets[1] & sets[2] & sets[3])
    pv = (np.sum(nd >= obs) + 1) / (n + 1)
    print(f"\n  Permutation (n={n})")
    print(f"    Observed: {obs}")
    print(f"    Null mean +/- sd: {nd.mean():.1f} +/- {nd.std():.1f}")
    print(f"    Null 95% CI: [{np.percentile(nd,2.5):.0f}, {np.percentile(nd,97.5):.0f}]")
    print(f"    p-value: {pv:.6f}")
    return {"observed": int(obs), "expected_mean": float(nd.mean()),
            "expected_sd": float(nd.std()),
            "ci_95_lower": float(np.percentile(nd,2.5)),
            "ci_95_upper": float(np.percentile(nd,97.5)),
            "p_value": float(pv), "n_permutations": n,
            "null_distribution": nd.tolist()}

# ====================== Annotate ================================
MECH = {"JUN":"UPR_Stress","FOS":"UPR_Stress","JUND":"UPR_Stress",
        "ATF4":"UPR_Stress","DDIT3":"UPR_Stress","XBP1":"UPR_Stress",
        "CALR":"UPR_Stress","HSPA5":"UPR_Stress","HSP90B1":"UPR_Stress",
        "ATF6":"UPR_Stress","CREB3":"UPR_Stress",
        "SPI1":"Myeloid_Bias","CEBPA":"Myeloid_Bias","CEBPB":"Myeloid_Bias",
        "CEBPD":"Myeloid_Bias","RUNX1":"Myeloid_Bias","IRF8":"Myeloid_Bias",
        "PAX5":"Lymphoid_Depletion","EBF1":"Lymphoid_Depletion",
        "TCF3":"Lymphoid_Depletion","TCF4":"Lymphoid_Depletion","TCF12":"Lymphoid_Depletion",
        "NFKB1":"NFkB_Signaling","NFKB2":"NFkB_Signaling","RELA":"NFkB_Signaling",
        "RELB":"NFkB_Signaling","TNFAIP3":"NFkB_Signaling",
        "NLRP3":"Inflammasome","IL1B":"Inflammasome","IL18":"Inflammasome",
        "PYCARD":"Inflammasome","CASP1":"Inflammasome",
        "RIPK1":"Necroptosis","RIPK3":"Necroptosis","MLKL":"Necroptosis","CASP8":"Necroptosis",
        "STAT1":"JAK_STAT","STAT3":"JAK_STAT","STAT5A":"JAK_STAT","STAT5B":"JAK_STAT",
        "IRF1":"Interferon","IFITM3":"Interferon","ISG15":"Interferon",
        "MPO":"NETosis","ELANE":"NETosis","PADI4":"NETosis","MMP8":"NETosis","MMP9":"NETosis",
        "BCL2":"Survival_Advantage","BCL2L1":"Survival_Advantage","MCL1":"Survival_Advantage",
        "GATA1":"Erythroid_Defect","BCL11A":"Erythroid_Defect","KLF1":"Erythroid_Defect",
        "S100A8":"Calprotectin","S100A9":"Calprotectin"}

EVID = {"SPI1":"Direct_Validation","PAX5":"Direct_Validation","CEBPB":"Direct_Validation",
        "MPO":"Direct_Validation","PADI4":"Direct_Validation","IL1B":"Direct_Validation",
        "IL18":"Direct_Validation","NFKB1":"Direct_Validation","RIPK1":"Direct_Validation",
        "RIPK3":"Direct_Validation","MLKL":"Direct_Validation",
        "JUN":"Indirect_Support","FOS":"Indirect_Support","JUND":"Indirect_Support",
        "STAT3":"Therapeutic_Evidence","STAT1":"Indirect_Support"}

def annotate(df):
    df["vexas_mechanism"] = df["gene"].map(MECH).fillna("Other")
    df["evidence_level"] = df["gene"].map(EVID).fillna("Computational_Prediction")
    return df

# ========================== Main ================================
def main():
    print("="*60)
    print("  Phase 1: Multi-dimensional Quantitative Gene Classification")
    print("="*60)

    print("\n[1] Loading driver gene data...")
    raw, meta = load_data()

    print("\n[2] [0,1] Normalization + z-score...")
    S, Z, MAX_all = normalize(raw)
    min_s = MIN_RAW / MAX_all
    print(f"  S-threshold = {MIN_RAW}/{MAX_all:.4f} = {min_s:.4f}")

    Z.to_csv(os.path.join(OUT_DIR, "Z_matrix.csv"), encoding="utf-8-sig")

    print("\n[3] CTS...")
    cts = compute_cts(S)
    print(f"  range [{cts.min():.4f},{cts.max():.4f}]")

    print("\n[4] HPS...")
    hps = compute_hps(S)
    print(f"  range [{hps.min():.4f},{hps.max():.4f}]")

    print("\n[5] Layer Consistency...")
    lc = compute_lc(S)
    print(f"  range [{lc.min():.4f},{lc.max():.4f}]")

    print("\n[6] Classification...")
    mask = (raw[COL_ORDER] > MIN_RAW).any(axis=1)
    Sf, lcf, ctsf, hpsf, Zf = S.loc[mask], lc[mask], cts[mask], hps[mask], Z.loc[mask]
    print(f"  Active genes: {mask.sum()}/{len(mask)}")

    cl = classify(Sf, lcf, min_s)
    cl["CTS"] = ctsf.values
    cl["HPS"] = hpsf.values
    cl["layer_consistency"] = lcf.values
    cl["propagation_depth"] = (Sf[COL_ORDER] > 0).sum(axis=1).values

    for c in COL_ORDER:
        cl[f"S_{c}"] = Sf[c].values
        cl[f"Z_{c}"] = Zf[c].values

    print("\n[7] Permutation test...")
    pr = perm_test(raw)

    print("\n[8] Annotation...")
    cl = annotate(cl)
    cl = cl.merge(meta, left_on="gene", right_index=True, how="left")

    out = os.path.join(OUT_DIR, "gene_cascade_classification.csv")
    cl.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n  [OK] {out}")

    for cat in ["Cascade_Core_Uniform","Cascade_Core_Decay","HSC_Initiator",
                "GMP_Propagator","Mono_Amplifier","Neut_Amplifier","Other"]:
        print(f"    {cat:<24}: {(cl['cascade_category']==cat).sum()}")

    pr["N_total_genes"] = len(cl)
    for cat in ["Cascade_Core_Uniform","Cascade_Core_Decay","HSC_Initiator",
                "GMP_Propagator","Mono_Amplifier","Neut_Amplifier"]:
        pr[f"N_{cat}"] = int((cl["cascade_category"]==cat).sum())
    core = cl[cl["cascade_category"].isin(["Cascade_Core_Uniform","Cascade_Core_Decay"])]
    pr["verified_rate"] = float((core["evidence_level"]!="Computational_Prediction").mean()) if len(core)>0 else 0
    with open(os.path.join(OUT_DIR, "cascade_propagation_stats.json"),"w",encoding="utf-8") as f:
        json.dump(pr, f, indent=2, ensure_ascii=False)

    # CTS verification
    ub = {"JUN","FOS","JUND","NFKB1","STAT3","SPI1"}
    sp = {"MPO","ELANE","PADI4"}
    ui = list(ub & set(cl["gene"])); si = list(sp & set(cl["gene"]))
    if len(ui)>=3 and len(si)>=3:
        _, p = mannwhitneyu(cl[cl["gene"].isin(ui)]["CTS"], cl[cl["gene"].isin(si)]["CTS"], alternative="less")
        print(f"\n  CTS verification: p={p:.4f}")

    print(f"\n{'='*60}\n  Phase 1 Complete\n{'='*60}")

if __name__=="__main__":
    main()
