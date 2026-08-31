"""
Caunagi Cell-Type-Specific Trajectory Driver Gene Identification
=====================================

Principle:
  Stage 1: Following the logic of Cluster_to_Celltype.py, use the celltype column in the h5ad file to determine
            the cell type corresponding to each Leiden cluster
  Stage 2: Following the logic of Map_Celltype.py, parse trajectory filenames in idremResults
            to determine the cell-type composition of each trajectory
  Stage 3: Extract trajectory-specific DREM TF and Dynamic Marker evidence for each trajectory,
            then aggregate by cell type and rank candidate driver genes for each cell type

Output:
  results/driver_genes/overall_driver_genes.csv           — integrated results across all cell types
  results/driver_genes/{celltype}_driver_genes.csv        — cell-type-specific results
  results/driver_genes/per_celltype_summary.csv           — Cross-Cell-Type Summary
  results/driver_genes/trajectory_celltype_mapping.csv    — trajectory-to-cell-type mapping table
"""

import os, sys, json, glob, pickle, re, math, warnings
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import rankdata

warnings.filterwarnings('ignore')

# ============================================================
# Part 1: Trajectory-to-Cell-Type Mapping (Based on Downstream Processing Directory Logic)
# ============================================================

def read_obs_column_h5ad(h5ad_path, col_name):
    """Read an obs column from h5ad (compatible with categorical and regular columns)"""
    import h5py
    with h5py.File(h5ad_path, 'r') as f:
        cat_path = f'obs/{col_name}/categories'
        codes_path = f'obs/{col_name}/codes'
        if cat_path in f and codes_path in f:
            cats = [x.decode('utf-8') if isinstance(x, bytes) else str(x)
                    for x in f[cat_path][:]]
            codes = f[codes_path][:]
            return [cats[c] if 0 <= c < len(cats) else str(c)
                    for c in codes]
        elif f'obs/{col_name}' in f:
            vals = f[f'obs/{col_name}'][:]
            return [x.decode('utf-8') if isinstance(x, bytes) else str(x)
                    for x in vals]
        return None


def create_cluster_celltype_mapping(temp_path, iteration, stages):
    """
    Create a {leiden_cluster_id: cell_type_name} mapping for each stage
    Based on the dominant cell type in each cluster (>50% is considered dominant)
    """
    mappings = {}
    for stage in range(stages):
        path = os.path.join(temp_path, str(iteration), 'stagedata', f'{stage}.h5ad')
        leiden_vals = read_obs_column_h5ad(path, 'leiden')
        celltype_vals = read_obs_column_h5ad(path, 'celltype')
        if leiden_vals is None or celltype_vals is None:
            continue

        # Count the cell-type composition of each cluster
        cluster_ct = defaultdict(lambda: defaultdict(int))
        for lei, ct in zip(leiden_vals, celltype_vals):
            cluster_ct[lei][ct] += 1

        # Determine the dominant cell type
        stage_map = {}
        print(f"\n  Stage {stage} cluster-to-cell-type mapping:")
        for cluster in sorted(cluster_ct.keys(), key=int):
            counts = cluster_ct[cluster]
            total = sum(counts.values())
            dominant_ct = max(counts, key=counts.get)
            dominant_pct = 100.0 * counts[dominant_ct] / total
            if dominant_pct >= 50:
                stage_map[cluster] = dominant_ct
            else:
                # Mixed cluster: use the top two cell types
                top2 = sorted(counts, key=counts.get, reverse=True)[:2]
                stage_map[cluster] = '/'.join(top2)
            print(f"    leiden_{cluster}: {stage_map[cluster]} ({dominant_pct:.0f}%, {total} cells)")
        mappings[stage] = stage_map
    return mappings


def parse_trajectory_to_celltype(traj_filename, stage_mappings):
    """
    Parse a trajectory filename such as '0-1n2.txt_viz' into a cell-type trajectory

    Returns: (trajectory_name, cell_type_path_string, primary_cell_type_list)
    Example: '0-1n2.txt_viz' → ('0-1n2', 'HSPC → Neutrophil+Monocyte', ['HSPC', 'Neutrophil', 'Monocyte'])
    """
    name = traj_filename.replace('.txt_viz', '')
    parts = name.split('-')
    if len(parts) != 2:
        return None

    # Stage 0 cluster
    s0_ct = stage_mappings[0].get(parts[0], parts[0])

    # Stage 1 clusters (may have multiple joined by 'n')
    s1_clusters = parts[1].split('n')
    s1_cts = [stage_mappings[1].get(c, c) for c in s1_clusters]
    s1_str = '+'.join(sorted(set(s1_cts)))

    # Construct the cell-type trajectory string
    ct_path = f"{s0_ct} → {s1_str}"

    # Collect all cell types involved in this trajectory (split mixed labels)
    all_cts = set()
    if '/' in s0_ct:
        all_cts.update(s0_ct.split('/'))
    else:
        all_cts.add(s0_ct)
    for c in s1_cts:
        if '/' in c:
            all_cts.update(c.split('/'))
        else:
            all_cts.add(c)
    all_cts = sorted(all_cts)

    return {
        'trajectory_name': name,
        'filename': traj_filename,
        'celltype_path': ct_path,
        'involved_celltypes': '+'.join(all_cts),
        'celltype_set': set(all_cts),
    }


# ============================================================
# Part 2: Per-Trajectory Evidence Extraction
# ============================================================

def robust_parse_drem_json(filepath):
    """Robust DREM.json parser"""
    tf_scores = {}
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return tf_scores

    start = content.find('data=')
    if start == -1:
        start = content.find('[')
        if start >= 0:
            inner = content[start:]
        else:
            return tf_scores
    else:
        inner = content[start:]
        eq_idx = inner.find('[')
        if eq_idx == -1:
            return tf_scores
        inner = inner[eq_idx:]

    for trim in range(0, 50):
        candidate = inner if trim == 0 else inner[:-trim]
        candidate = re.sub(r',\s*\]\s*$', ']', candidate)
        candidate = re.sub(r',\s*null\s*,?\s*\]\s*$', ']', candidate)
        candidate = candidate.rstrip().rstrip(';').rstrip()
        try:
            data = json.loads(candidate, strict=False)
            break
        except json.JSONDecodeError:
            continue
    else:
        return tf_scores

    if not isinstance(data, list) or len(data) == 0:
        return tf_scores

    nodes = data[0]
    for node in nodes:
        if not isinstance(node, dict):
            continue
        for tf_entry in node.get("ETF", []):
            try:
                tf_name = tf_entry[0].split()[0].upper()
                pval = None
                for pos in [6, 5, 4, -1, -2]:
                    try:
                        val = float(tf_entry[pos])
                        if 0 < val <= 1:
                            pval = val
                            break
                    except (ValueError, IndexError):
                        continue
                if pval is not None and pval > 0:
                    score = -math.log10(pval)
                else:
                    try:
                        score = float(tf_entry[4])
                    except (ValueError, IndexError):
                        continue
                tf_scores[tf_name] = max(tf_scores.get(tf_name, 0), score)
            except (IndexError, ValueError):
                continue
    return tf_scores


def extract_per_trajectory_evidence(temp_path, final_iteration):
    """
    Extract evidence separately for each trajectory:
      - DREM TFs (trajectory-specific)
      - Dynamic markers (trajectory-specific)
    Returns: {trajectory_name: {'DREM': {tf: score}, 'dynamic': {gene: score}, 'dynamic_detail': {gene: info}}}
    """
    idrem_dir = os.path.join(temp_path, str(final_iteration), 'idremResults')
    trajectory_evidence = {}

    # Load dynamic_markers (already organized by trajectory)
    dm_path = os.path.join(temp_path, 'dynamic_markers.pkl')
    dm_data = {}
    if os.path.exists(dm_path):
        with open(dm_path, 'rb') as f:
            dm_data = pickle.load(f)

    for entry in os.listdir(idrem_dir):
        if not entry.endswith('.txt_viz'):
            continue
        traj_name = entry.replace('.txt_viz', '')
        drem_path = os.path.join(idrem_dir, entry, 'DREM.json')
        traj_ev = {'DREM': {}, 'dynamic': {}, 'dynamic_detail': {}}

        # DREM TFs
        if os.path.exists(drem_path):
            traj_ev['DREM'] = robust_parse_drem_json(drem_path)

        # Dynamic markers (trajectory-specific)
        if traj_name in dm_data:
            for direction in ['increasing', 'decreasing']:
                if direction in dm_data[traj_name]:
                    td = dm_data[traj_name][direction]
                    gd = td.get('gene', {})
                    qv = td.get('qval', {})
                    fc = td.get('log2fc', {})
                    for gi in gd:
                        gene = str(gd[gi]).upper()
                        q = float(qv.get(gi, 1.0))
                        f = float(fc.get(gi, 0.0))
                        dyn_score = (-math.log10(max(q, 1e-300))) * math.log2(1 + abs(f))
                        if gene not in traj_ev['dynamic'] or dyn_score > traj_ev['dynamic'][gene]:
                            traj_ev['dynamic'][gene] = dyn_score
                        traj_ev['dynamic_detail'][gene] = {
                            'qval': q, 'log2fc': f, 'direction': direction
                        }

        trajectory_evidence[traj_name] = traj_ev

    return trajectory_evidence


# ============================================================
# Part 3: General Scoring Utilities
# ============================================================

def normalize_dict(d):
    if not d:
        return {}
    vals = np.array(list(d.values()))
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return {k: 0.5 for k in d}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in d.items()}


# Gene-category system (consistent with driver_gene_identification.py)
_KNOWN_TFS = {
    'SPI1', 'PAX5', 'GATA1', 'GATA2', 'GATA3', 'CEBPA', 'CEBPB', 'CEBPD', 'CEBPE', 'CEBPG',
    'RUNX1', 'RUNX2', 'RUNX3', 'IRF1', 'IRF2', 'IRF3', 'IRF4', 'IRF5', 'IRF6', 'IRF7', 'IRF8', 'IRF9',
    'EBF1', 'TCF3', 'TCF4', 'TCF12', 'JUN', 'JUNB', 'JUND', 'FOS', 'FOSB', 'FOSL1', 'FOSL2',
    'STAT1', 'STAT2', 'STAT3', 'STAT4', 'STAT5A', 'STAT5B', 'STAT6',
    'NFKB1', 'NFKB2', 'REL', 'RELA', 'RELB',
    'KLF1', 'KLF2', 'KLF3', 'KLF4', 'KLF5', 'KLF6', 'MYC', 'MYB', 'MYCN', 'MAX', 'MXD1', 'MXI1',
    'EGR1', 'EGR2', 'EGR3', 'EGR4', 'ZEB1', 'ZEB2', 'SNAI1', 'SNAI2', 'TWIST1', 'TWIST2',
    'BCL11A', 'BCL11B', 'BCL6', 'PRDM1', 'TBX21',
    'FOXP3', 'RORC', 'AHR', 'HIF1A', 'EPAS1', 'ARNT',
    'ETV1', 'ETV2', 'ETV3', 'ETV4', 'ETV5', 'ETV6', 'ETV7', 'ERG', 'FLI1',
    'SOX4', 'SOX5', 'SOX6', 'SOX9', 'SOX10',
    'CTCF', 'YY1', 'RAD21', 'SMC1A', 'SMC3', 'TAF1', 'TBP',
    'SMAD1', 'SMAD2', 'SMAD3', 'SMAD4', 'SMAD5',
    'MEF2A', 'MEF2B', 'MEF2C', 'MEF2D', 'SRF', 'ELK1',
    'NR3C1', 'NR3C2', 'ESR1', 'ESR2', 'AR', 'PPARG', 'VDR', 'RARA',
    'FOXA1', 'FOXA2', 'FOXA3', 'FOXO1', 'FOXO3', 'FOXO4', 'FOXM1',
    'PBX1', 'PBX2', 'PBX3', 'PBX4', 'MEIS1', 'MEIS2', 'MEIS3', 'HOXA9',
    'TAL1', 'LYL1', 'LMO2', 'HEY1', 'HES1', 'NOTCH1', 'RBPJ',
    'NFE2', 'NFE2L2', 'BACH1', 'BACH2', 'MAF', 'MAFB', 'MAFF', 'MAFG', 'MAFK',
    'XBP1', 'ATF4', 'ATF6', 'DDIT3', 'CREB1',
    'ZBTB16', 'GFI1', 'GFI1B',
    'MITF', 'TFEB', 'TFE3', 'USF1', 'USF2',
    'E2F1', 'E2F2', 'E2F3', 'E2F4', 'HMGA1', 'HMGA2', 'SALL4', 'WT1',
    'CUX1', 'KMT2A', 'KMT2D', 'EZH2', 'SUZ12', 'JARID2',
}

_HEMOGLOBIN_GENES = {'HBA1', 'HBA2', 'HBB', 'HBD', 'HBE1', 'HBG1', 'HBG2', 'HBZ', 'HBQ1', 'HBM'}
_RIBOSOMAL_PREFIXES = ('RPS', 'RPL', 'RPP', 'MRPS', 'MRPL')
_IG_PREFIXES = ('IGK', 'IGL', 'IGH', 'IGJ', 'IGLL', 'IGLC', 'IGKC', 'IGHA', 'IGHG', 'IGHM', 'IGHD', 'IGHE')
_MITO_PREFIXES = ('MT-', 'MTRNR', 'ATP6', 'ATP8', 'COX1', 'COX2', 'COX3', 'CYTB', 'ND1', 'ND2', 'ND3', 'ND4', 'ND5', 'ND6')


def classify_gene_category(gene):
    g = gene.upper()
    if g in _KNOWN_TFS:
        return 'tf'
    if g in _HEMOGLOBIN_GENES:
        return 'hemoglobin'
    if any(g.startswith(p) for p in _RIBOSOMAL_PREFIXES):
        return 'ribosomal'
    if any(g.startswith(p) for p in _IG_PREFIXES):
        return 'immunoglobulin'
    if any(g.startswith(p) for p in _MITO_PREFIXES):
        return 'mitochondrial'
    return 'other'


def category_multiplier(gene):
    cat = classify_gene_category(gene)
    if cat == 'hemoglobin':
        return 0.45
    elif cat == 'ribosomal':
        return 0.55
    elif cat == 'immunoglobulin':
        return 0.55
    elif cat == 'mitochondrial':
        return 0.70
    elif cat == 'tf':
        return 1.30
    return 1.00


# ============================================================
# Part 4: Cell-Type-Specific Driver Gene Scoring
# ============================================================

def compute_per_celltype_drivers(traj_evidence, traj_celltype_map):
    """
    For each cell type, aggregate evidence from all associated trajectories and calculate driver-gene scores

    Strategy:
      - If a trajectory involves multiple cell types, it contributes evidence to each involved cell type
      - For a given cell type, use the maximum DREM TF score across all relevant trajectories
      - Use the maximum Dynamic Marker score across all relevant trajectories
    """
    # Collect gene evidence for each cell type
    ct_genes = defaultdict(lambda: {'DREM': defaultdict(list), 'dynamic': defaultdict(list),
                                      'traj_count': set()})

    for traj_name, evidence in traj_evidence.items():
        if traj_name not in traj_celltype_map:
            continue
        celltypes = traj_celltype_map[traj_name]['celltype_set']

        for ct in celltypes:
            ct_genes[ct]['traj_count'].add(traj_name)

            # DREM TFs
            for gene, score in evidence.get('DREM', {}).items():
                ct_genes[ct]['DREM'][gene].append(score)

            # Dynamic markers
            for gene, score in evidence.get('dynamic', {}).items():
                ct_genes[ct]['dynamic'][gene].append(score)

    # Calculate scores for each cell type
    results = {}
    for ct in sorted(ct_genes.keys()):
        ct_data = ct_genes[ct]
        n_traj = len(ct_data['traj_count'])

        # Aggregate: DREM → maximum score; dynamic → maximum score
        drem_agg = {g: max(scores) for g, scores in ct_data['DREM'].items()}
        dyn_agg = {g: max(scores) for g, scores in ct_data['dynamic'].items()}

        # Normalize
        norm_drem = normalize_dict(drem_agg)
        norm_dyn = normalize_dict(dyn_agg)

        # All relevant genes
        all_genes = set(drem_agg.keys()) | set(dyn_agg.keys())

        gene_scores = []
        for gene in all_genes:
            drem = norm_drem.get(gene, 0)
            dyn = norm_dyn.get(gene, 0)
            ev_count = (1 if drem > 0 else 0) + (1 if dyn > 0 else 0)

            score = 0.60 * drem + 0.40 * dyn
            score *= category_multiplier(gene)

            gene_scores.append({
                'gene': gene,
                'celltype': ct,
                'driver_score': round(score, 6),
                'evidence_count': ev_count,
                'n_trajectories': n_traj,
                'gene_category': classify_gene_category(gene),
                'DREM_score': round(drem, 4),
                'dynamic_score': round(dyn, 4),
            })

        df = pd.DataFrame(gene_scores)
        if len(df) > 0:
            df = df.sort_values('driver_score', ascending=False).reset_index(drop=True)
            df['rank'] = range(1, len(df) + 1)
        results[ct] = df

    return results


# ============================================================
# Part 5: Main Entry Point
# ============================================================

def main(temp_path, final_iteration, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 70)
    print("  Caunagi Cell-Type-Specific Trajectory Driver Gene Identification")
    print("=" * 70)
    print(f"  Data directory:   {temp_path}")
    print(f"  Final iteration:   {final_iteration}")
    print(f"  Output directory:   {output_dir}")
    print("=" * 70)

    # ---- Step 1: Build cluster-to-cell-type mapping ----
    print("\n[Step 1] Building cluster-to-cell-type mapping...")
    stages = 2  # VEXAS data: two stages, healthy and disease
    stage_mappings = create_cluster_celltype_mapping(temp_path, final_iteration, stages)

    # ---- Step 2: Parse all trajectories into cell types ----
    print("\n[Step 2] Parsing trajectories → cell types...")
    idrem_dir = os.path.join(temp_path, str(final_iteration), 'idremResults')
    traj_celltype_map = {}
    mapping_rows = []

    for entry in sorted(os.listdir(idrem_dir)):
        if not entry.endswith('.txt_viz'):
            continue
        result = parse_trajectory_to_celltype(entry, stage_mappings)
        if result:
            traj_celltype_map[result['trajectory_name']] = result
            mapping_rows.append({
                'trajectory': result['trajectory_name'],
                'filename': result['filename'],
                'celltype_path': result['celltype_path'],
                'involved_celltypes': result['involved_celltypes'],
            })
            print(f"  {result['trajectory_name']:<15}  →  {result['celltype_path']}")

    # Save mapping table
    mapping_df = pd.DataFrame(mapping_rows)
    mapping_df.to_csv(os.path.join(output_dir, 'trajectory_celltype_mapping.csv'), index=False)
    print(f"\n  Mapping table saved: trajectory_celltype_mapping.csv")

    # ---- Step 3: Extract trajectory-specific evidence ----
    print("\n[Step 3] Extracting DREM + Dynamic evidence for each trajectory...")
    traj_evidence = extract_per_trajectory_evidence(temp_path, final_iteration)
    print(f"  Extraction completed for {len(traj_evidence)} trajectories")

    # ---- Step 4: Calculate driver genes by cell type ----
    print("\n[Step 4] Aggregating and scoring by cell type...")
    ct_results = compute_per_celltype_drivers(traj_evidence, traj_celltype_map)

    # ---- Step 5: Output ----
    print("\n[Step 5] Saving results...")

    # 5a. Save each cell type separately
    summary_rows = []
    for ct in sorted(ct_results.keys()):
        df = ct_results[ct]
        if len(df) == 0:
            continue
        safe_name = ct.replace(' ', '_').replace('/', '_')
        csv_path = os.path.join(output_dir, f'{safe_name}_driver_genes.csv')
        df.to_csv(csv_path, index=False)
        print(f"  {ct:<20} → {len(df)} candidate genes → {safe_name}_driver_genes.csv")

        top_tfs = df[df['gene_category'] == 'tf'].head(5)['gene'].tolist()
        summary_rows.append({
            'celltype': ct,
            'candidate_count': len(df),
            'top5_TFs': ', '.join(top_tfs) if top_tfs else 'N/A',
            'n_trajectories': df['n_trajectories'].iloc[0] if len(df) > 0 else 0,
        })

    # 5b. Cross-Cell-Type Summary
    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(output_dir, 'per_celltype_summary.csv'), index=False)

    # 5c. Copy integrated results
    overall_src = os.path.join(output_dir, 'driver_gene_candidates.csv')
    if not os.path.exists(overall_src):
        overall_src = os.path.join(temp_path, 'driver_results', 'driver_gene_candidates.csv')
    if os.path.exists(overall_src):
        import shutil
        shutil.copy(overall_src, os.path.join(output_dir, 'overall_driver_genes.csv'))
        print(f"\n  Integrated results copied: overall_driver_genes.csv")

    # ---- Terminal output ----
    print("\n" + "=" * 70)
    print("  Cross-Cell-Type Summary")
    print("=" * 70)
    print(f"{'CellType':<20} {'Candidates':<12} {'Traj':<6} {'Top-5 TFs'}")
    print("-" * 70)
    for row in summary_rows:
        print(f"{row['celltype']:<20} {row['candidate_count']:<12} {row['n_trajectories']:<6} {row['top5_TFs'][:60]}")

    # Print detailed Top 10 results for priority cell types
    print("\n" + "=" * 70)
    print("  Top 10 Candidate Genes for Priority Cell Types")
    print("=" * 70)
    priority_cts = ['Neutrophil', 'Monocyte', 'HSPC', 'CD4+ T', 'CD8+ T', 'B']
    for ct in priority_cts:
        if ct not in ct_results:
            continue
        df = ct_results[ct]
        top10 = df.head(10)
        print(f"\n  [{ct}]  ({len(df)} candidates)")
        print(f"  {'Rank':<5} {'Gene':<12} {'Score':<9} {'Cat':<12}")
        print(f"  {'-'*40}")
        for _, r in top10.iterrows():
            print(f"  {int(r['rank']):<5} {r['gene']:<12} {r['driver_score']:.4f}   {r['gene_category']:<12}")

    print(f"\nAll results saved to: {output_dir}")
    return ct_results


if __name__ == '__main__':
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    TEMP_PATH = os.path.join(PROJECT_ROOT, 'temp_path')
    FINAL_ITERATION = 5
    OUTPUT_DIR = SCRIPT_DIR

    ct_results = main(TEMP_PATH, FINAL_ITERATION, OUTPUT_DIR)
