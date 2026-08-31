"""
Caunagi Disease Driver Gene Identification — Complete Five-Dimensional Evidence Integration Framework

Integrates all Caunagi outputs:
  1. geneWeight (iterative gene weights)         → 40% weight
  2. DREM TF Score (iDREM TF activity)      → 25% weight
  3. Dynamic Markers (dynamic markers)       → 20% weight
  4. Network topology (MFVS + centrality)          → 10% weight
  5. HC Markers (hierarchical-clustering markers)        →  5% weight

Output: candidate driver genes ranked by driver_score in descending order (CSV + statistical report)
"""

import os, json, glob, pickle, re, math, sys
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from scipy.stats import rankdata

# ============================================================
# Part 1: Utility Functions
# ============================================================

def robust_parse_drem_json(filepath):
    """
    Robust DREM.json parser.
    Returns: {tf_name: -log10(p_value)} dictionary
    """
    tf_scores = {}
    try:
        with open(filepath, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
    except Exception:
        return tf_scores

    # Look for data=[...] or data = [...]
    start = content.find('data=')
    if start == -1:
        # Try content that starts directly with [
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

    # Progressively trim trailing invalid characters
    for trim in range(0, 50):
        candidate = inner if trim == 0 else inner[:-trim]
        # Remove malformed trailing comma before ] 
        candidate = re.sub(r',\s*\]\s*$', ']', candidate)
        candidate = re.sub(r',\s*null\s*,?\s*\]\s*$', ']', candidate)
        # Remove trailing semicolon
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
                # Check multiple possible p-value positions (positions vary across iDREM versions)
                pval = None
                for pos in [6, 5, 4, -1, -2]:
                    try:
                        val = float(tf_entry[pos])
                        if 0 < val <= 1:  # p-value should be in the (0, 1] range
                            pval = val
                            break
                    except (ValueError, IndexError):
                        continue
                if pval is not None and pval > 0:
                    score = -math.log10(pval)
                else:
                    # Fallback: directly use the value at position 4 as the score
                    try:
                        score = float(tf_entry[4])
                    except (ValueError, IndexError):
                        continue
                tf_scores[tf_name] = max(tf_scores.get(tf_name, 0), score)
            except (IndexError, ValueError):
                continue
    return tf_scores


def normalize_dict(d):
    """Min-max normalize to [0, 1]"""
    if not d:
        return {}
    vals = np.array(list(d.values()))
    vmin, vmax = vals.min(), vals.max()
    if vmax == vmin:
        return {k: 0.5 for k in d}
    return {k: float((v - vmin) / (vmax - vmin)) for k, v in d.items()}


def load_gene_names(h5ad_path):
    """Read the gene-name list from an h5ad file (use h5py to avoid a Scanpy dependency)"""
    try:
        import h5py
        with h5py.File(h5ad_path, 'r') as f:
            for key in ['var/_index', 'var/index', 'var/name', 'var/gene_symbols']:
                if key in f:
                    return [x.decode('utf-8') if isinstance(x, bytes) else str(x)
                            for x in f[key][:]]
            # Finally, try any suitable column under var
            if 'var' in f:
                for k in f['var']:
                    if 'name' in k.lower() or 'gene' in k.lower() or 'symbol' in k.lower() or '_index' in k.lower():
                        vals = f[f'var/{k}'][:]
                        return [x.decode('utf-8') if isinstance(x, bytes) else str(x) for x in vals]
        return None
    except ImportError:
        try:
            import scanpy as sc
            adata = sc.read_h5ad(h5ad_path)
            return list(adata.var.index)
        except ImportError:
            return None


# ============================================================
# Part 2: Evidence Dimension Extraction
# ============================================================

class CaunagiEvidenceCollector:
    """
    Collect all five evidence dimensions from the Caunagi output directory
    """

    def __init__(self, temp_path, final_iteration, species='human'):
        self.temp_path = temp_path
        self.iter_num = final_iteration
        self.species = species
        self.gene_names = None
        self.evidence = {}

    # ---------- Evidence 1: geneWeight (iterative gene weights) ----------
    def extract_gene_weight(self):
        """Read the geneWeight layer from each stage h5ad in the final iteration and aggregate it into gene scores"""
        print("[1/5] Extracting geneWeight ...")
        gene_records = defaultdict(list)

        # Read gene names
        stage0_path = os.path.join(self.temp_path, str(self.iter_num),
                                   'stagedata', '0.h5ad')
        self.gene_names = load_gene_names(stage0_path)
        if self.gene_names is None:
            print("  Warning: unable to read the gene-name list; skipping geneWeight")
            return {}

        import h5py
        for stage in range(10):
            fpath = os.path.join(self.temp_path, str(self.iter_num),
                                'stagedata', f'{stage}.h5ad')
            if not os.path.exists(fpath):
                break
            try:
                mean_weights = self._read_gene_weight_sparse_mean(fpath)
                if mean_weights is None:
                    continue
                nonzero = np.where(mean_weights > 0)[0]
                n_found = 0
                for col_idx in nonzero:
                    if col_idx < len(self.gene_names):
                        gene = self.gene_names[col_idx].upper()
                        gene_records[gene].append(float(mean_weights[col_idx]))
                        n_found += 1
                print(f"  Stage {stage}: {n_found} genes with nonzero weights")
            except Exception as e:
                print(f"  Failed to read stage {stage}: {e}")

        # Calculate aggregate score: mean_weight * (1 - CV)
        scores = {}
        for gene, weights in gene_records.items():
            mean_w = np.mean(weights)
            if len(weights) > 1 and mean_w > 0:
                cv = np.std(weights) / mean_w
            else:
                cv = 0.0
            scores[gene] = mean_w * (1.0 - min(float(cv), 1.0))
        print(f"  -> Extracted geneWeight scores for {len(scores)} genes")
        self.evidence['geneWeight'] = scores
        return scores

    @staticmethod
    def _read_gene_weight_sparse_mean(h5ad_path):
        """Memory-efficient calculation of geneWeight column means"""
        import h5py
        with h5py.File(h5ad_path, 'r') as f:
            if 'layers/geneWeight' not in f:
                return None
            gw = f['layers/geneWeight']
            if 'data' not in gw or len(gw['data']) == 0:
                return None
            data = gw['data'][:]
            indices = gw['indices'][:]
            indptr = gw['indptr'][:]
            shape = gw.attrs.get('shape', None)
            if shape is None:
                n_rows = len(indptr) - 1
                n_cols = int(indices.max()) + 1 if len(indices) > 0 else 0
            else:
                n_rows, n_cols = int(shape[0]), int(shape[1])
            # Accumulate column sums directly to avoid constructing a dense matrix
            col_sums = np.zeros(n_cols, dtype=np.float64)
            for row_idx in range(n_rows):
                start, end = indptr[row_idx], indptr[row_idx + 1]
                for j in range(start, end):
                    col_sums[indices[j]] += data[j]
            if n_rows > 0:
                mean_weights = col_sums / n_rows
            else:
                mean_weights = col_sums
            return mean_weights

    # ---------- Evidence 2: DREM TF Score ----------
    def extract_drem_scores(self):
        """Aggregate DREM TF scores across all trajectories and iterations"""
        print("[2/5] Extracting DREM TF scores ...")
        all_tfs = {}
        trajectory_tf_counts = defaultdict(int)  # TF → number of trajectories in which it appears

        for it in range(self.iter_num + 1):
            pattern = os.path.join(self.temp_path, str(it),
                                   "idremResults", "*", "DREM.json")
            drem_files = glob.glob(pattern)
            if not drem_files:
                continue
            for fp in drem_files:
                tfs = robust_parse_drem_json(fp)
                for tf, score in tfs.items():
                    all_tfs[tf] = max(all_tfs.get(tf, 0), score)
                    trajectory_tf_counts[tf] += 1

        # Integrated score = -log10(p) * log(number of trajectories + 1), accounting for cross-trajectory consistency
        integrated = {}
        for tf, score in all_tfs.items():
            n_traj = trajectory_tf_counts.get(tf, 1)
            integrated[tf] = score * math.log2(1 + n_traj)

        print(f"  -> Extracted {len(integrated)} DREM TFs across {len(drem_files) if drem_files else 0} trajectories")
        self.evidence['DREM'] = integrated
        self.evidence['DREM_trajectory_count'] = dict(trajectory_tf_counts)
        return integrated

    # ---------- Evidence 3: Dynamic Markers ----------
    def extract_dynamic_markers(self):
        """Extract dynamic markers: significance * effect_size * consistency"""
        print("[3/5] Extracting dynamic markers ...")
        pkl_path = os.path.join(self.temp_path, 'dynamic_markers.pkl')
        if not os.path.exists(pkl_path):
            print(f"  Warning: {pkl_path} was not found")
            self.evidence['dynamic'] = {}
            return {}

        with open(pkl_path, 'rb') as f:
            dm = pickle.load(f)

        gene_info = {}
        for track_name, track_data in dm.items():
            for direction in ['increasing', 'decreasing']:
                if direction not in track_data:
                    continue
                td = track_data[direction]
                genes_dict = td.get('gene', {})
                qvals = td.get('qval', {})
                log2fc = td.get('log2fc', {})
                for gi in genes_dict:
                    gene = str(genes_dict[gi]).upper()
                    qv = float(qvals.get(gi, 1.0))
                    fc = float(log2fc.get(gi, 0.0))
                    if gene not in gene_info:
                        gene_info[gene] = {
                            'min_qval': qv, 'max_log2fc': abs(fc),
                            'track_count': 1, 'direction': direction,
                            'tracks': [track_name]
                        }
                    else:
                        gi_ = gene_info[gene]
                        gi_['min_qval'] = min(gi_['min_qval'], qv)
                        gi_['max_log2fc'] = max(gi_['max_log2fc'], abs(fc))
                        gi_['track_count'] += 1
                        gi_['tracks'].append(track_name)

        scores = {}
        for gene, info in gene_info.items():
            qterm = -math.log10(max(info['min_qval'], 1e-300))
            fcterm = math.log2(1 + info['max_log2fc'])
            trajterm = 1 + info['track_count'] * 0.1
            scores[gene] = qterm * fcterm * trajterm

        print(f"  -> Extracted {len(scores)} dynamic markers")
        self.evidence['dynamic'] = scores
        self.evidence['dynamic_detail'] = gene_info
        return scores

    # ---------- Evidence 4: Network Topology (MFVS + centrality) ----------
    def extract_network_evidence(self, prior_net_path):
        """Construct a disease-specific network and calculate MFVS and network-centrality metrics"""
        print("[4/5] Constructing the disease-specific network and calculating topology metrics ...")

        # 4a. Load prior network
        if not os.path.exists(prior_net_path):
            print(f"  Warning: prior network {prior_net_path} was not found; skipping network analysis")
            self.evidence['network'] = {}
            return {}

        prior_net = pd.read_csv(prior_net_path)
        if 'from' not in prior_net.columns or 'to' not in prior_net.columns:
            print("  Warning: the prior network is missing the from/to columns")
            self.evidence['network'] = {}
            return {}

        prior_net = prior_net[['from', 'to']].dropna()
        prior_net['from'] = prior_net['from'].str.upper()
        prior_net['to'] = prior_net['to'].str.upper()

        # 4b. Collect disease-related gene set
        drem_tfs = set(self.evidence.get('DREM', {}).keys())
        dynamic_genes = set(self.evidence.get('dynamic', {}).keys())
        gw_genes = set(self.evidence.get('geneWeight', {}).keys())
        disease_genes = drem_tfs | dynamic_genes | gw_genes

        # 4c. Lenient filtering: the source TF must be a DREM TF and the target must be a disease-related gene
        mask = prior_net['from'].isin(drem_tfs) & prior_net['to'].isin(disease_genes)
        disease_net = prior_net[mask].copy()

        if len(disease_net) < 3:
            print(f"  Warning: too few network edges remain after filtering ({len(disease_net)}); expanding the filtering criteria")
            mask = (prior_net['from'].isin(disease_genes) &
                    prior_net['to'].isin(disease_genes))
            disease_net = prior_net[mask].copy()

        print(f"  Prior network: {len(prior_net)} edges → disease network: {len(disease_net)} edges")

        G = nx.from_pandas_edgelist(disease_net, source='from', target='to',
                                     create_using=nx.DiGraph())
        print(f"  Network nodes: {G.number_of_nodes()}, edges: {G.number_of_edges()}")

        # 4d. MFVS algorithm
        mfvs_set = self._mfvs_greedy(G)

        # 4e. Network centrality (performance and compatibility for large graphs)
        betweenness = {}
        try:
            # NetworkX >= 2.6 supports seed; <= 2.5 uses k for approximation
            if G.number_of_nodes() > 500:
                betweenness = nx.betweenness_centrality(G, k=min(200, G.number_of_nodes()))
            else:
                betweenness = nx.betweenness_centrality(G)
        except Exception:
            betweenness = {}
        pagerank = nx.pagerank(G, alpha=0.85) if G.number_of_nodes() > 0 else {}
        out_degree = dict(G.out_degree()) if G.number_of_nodes() > 0 else {}

        # 4f. Integrate network scores
        norm_between = normalize_dict(betweenness)
        norm_pr = normalize_dict(pagerank)
        norm_outdeg = normalize_dict(out_degree)

        all_nodes = set(betweenness.keys()) | set(pagerank.keys()) | set(out_degree.keys())
        network_scores = {}
        for node in all_nodes:
            bt = norm_between.get(node, 0)
            pr = norm_pr.get(node, 0)
            od = norm_outdeg.get(node, 0)
            is_mfvs = 1.0 if node in mfvs_set else 0.0
            network_scores[node] = 0.4 * bt + 0.3 * pr + 0.2 * od + 0.1 * is_mfvs

        print(f"  -> MFVS identified {len(mfvs_set)} feedback vertices")
        print(f"  -> Integrated network-topology scores for {len(network_scores)} nodes")

        self.evidence['network'] = network_scores
        self.evidence['MFVS_set'] = mfvs_set
        return network_scores

    def _mfvs_greedy(self, G):
        """Greedy MFVS algorithm"""
        working = G.copy()
        mfvs = set()

        for node in list(nx.nodes_with_selfloops(working)):
            mfvs.add(node)
            working.remove_node(node)

        while not nx.is_directed_acyclic_graph(working) and working.number_of_nodes() > 0:
            scores = {}
            for node in working.nodes():
                in_d = working.in_degree(node)
                out_d = working.out_degree(node)
                scores[node] = (in_d * out_d) + in_d + out_d
            best = max(scores, key=scores.get)
            mfvs.add(best)
            working.remove_node(best)
            working.remove_nodes_from(list(nx.isolates(working)))
        return mfvs

    # ---------- Evidence 5: HC Markers ----------
    def extract_hc_markers(self):
        """Extract hierarchical-clustering markers (as a binary bonus indicator), compatible with multiple pandas versions"""
        print("[5/5] Extracting HC markers ...")
        pkl_path = os.path.join(self.temp_path, 'hcmarkers.pkl')
        if not os.path.exists(pkl_path):
            print(f"  Warning: {pkl_path} was not found")
            self.evidence['HC'] = set()
            return set()

        hc = None
        # Try multiple loading methods
        errors = []
        for method_name, load_fn in [
            ('pickle.load', lambda: pickle.load(open(pkl_path, 'rb'))),
            ('pickle encoding=latin1', lambda: pickle.loads(open(pkl_path, 'rb').read(), encoding='latin1')),
            ('pickle encoding=bytes', lambda: pickle.loads(open(pkl_path, 'rb').read(), encoding='bytes')),
            ('pd.read_pickle', lambda: pd.read_pickle(pkl_path)),
        ]:
            try:
                hc = load_fn()
                if isinstance(hc, dict):
                    break
            except Exception as e:
                errors.append(f"{method_name}: {str(e)[:60]}")
                continue

        if hc is None or not isinstance(hc, dict):
            print("  HC loading failed: all methods failed")
            for err in errors:
                print(f"    {err}")
            self.evidence['HC'] = set()
            return set()

        hc_genes = set()
        try:
            for stage_val in hc.values():
                markers = stage_val.get('markers')
                if not isinstance(markers, dict):
                    continue
                for cluster_val in markers.values():
                    if not isinstance(cluster_val, dict):
                        continue
                    for level_val in cluster_val.values():
                        chosen = level_val.get('chosen')
                        if not isinstance(chosen, dict):
                            continue
                        for df in chosen.values():
                            if hasattr(df, 'index'):
                                genes = [str(g).upper() for g in df.index]
                                hc_genes.update(genes)
        except Exception as e:
            print(f"  HC parsing warning: {e}")

        print(f"  -> Extracted {len(hc_genes)} HC markers")
        self.evidence['HC'] = hc_genes
        return hc_genes

    # ============================================================
    # Main Workflow: Collect All Evidence
    # ============================================================
    def collect_all(self, prior_net_path=None):
        self.extract_gene_weight()
        self.extract_drem_scores()
        self.extract_dynamic_markers()
        if prior_net_path:
            self.extract_network_evidence(prior_net_path)
        else:
            print("[4/5] Skipping network analysis (prior_net_path was not provided)")
            self.evidence['network'] = {}
        self.extract_hc_markers()


# ============================================================
# Part 3: Multi-Evidence Integrated Scoring
# ============================================================

# ============================================================
# Gene classifier: distinguish regulators from downstream effector genes
# ============================================================

# Known human transcription-factor list (HGNC symbols, uppercase)
_KNOWN_TFS = {
    'SPI1','PAX5','GATA1','GATA2','GATA3','CEBPA','CEBPB','CEBPD','CEBPE','CEBPG',
    'RUNX1','RUNX2','RUNX3','IRF1','IRF2','IRF3','IRF4','IRF5','IRF6','IRF7','IRF8','IRF9',
    'EBF1','TCF3','TCF4','TCF12','JUN','JUNB','JUND','FOS','FOSB','FOSL1','FOSL2',
    'STAT1','STAT2','STAT3','STAT4','STAT5A','STAT5B','STAT6',
    'NFKB1','NFKB2','REL','RELA','RELB',
    'KLF1','KLF2','KLF3','KLF4','KLF5','KLF6','MYC','MYB','MYCN','MAX','MXD1','MXI1',
    'EGR1','EGR2','EGR3','EGR4','ZEB1','ZEB2','SNAI1','SNAI2','TWIST1','TWIST2',
    'PU1','BCL11A','BCL11B','BCL6','PRDM1','TBX21','GATA4','GATA5','GATA6',
    'FOXP3','RORC','AHR','HIF1A','EPAS1','ARNT','ARNT2','NPAS1','NPAS2','NPAS3','NPAS4',
    'ETV1','ETV2','ETV3','ETV4','ETV5','ETV6','ETV7','ERG','FLI1',
    'SOX2','SOX4','SOX5','SOX6','SOX9','SOX10','SOX17','SOX18',
    'CTCF','YY1','RAD21','SMC1A','SMC3','TAF1','TBP','TBPL1','TBPL2',
    'SMAD1','SMAD2','SMAD3','SMAD4','SMAD5','SMAD6','SMAD7','SMAD9',
    'MEF2A','MEF2B','MEF2C','MEF2D','SRF','ELK1','ELK3','ELK4',
    'NR3C1','NR3C2','ESR1','ESR2','AR','PGR','RARA','RARB','RARG',
    'RXRG','PPARG','VDR','THRA','THRB','HNF4A','NR1H4','NR5A1','NR5A2',
    'FOXA1','FOXA2','FOXA3','FOXO1','FOXO3','FOXO4','FOXM1',
    'PBX1','PBX2','PBX3','PBX4','MEIS1','MEIS2','MEIS3','HOXA9','HOXB4',
    'TAL1','TAL2','LYL1','LMO2','HEY1','HEY2','HES1','HES5','NOTCH1','RBPJ',
    'NFE2','NFE2L2','BACH1','BACH2','MAF','MAFB','MAFF','MAFG','MAFK',
    'XBP1','ATF4','ATF6','DDIT3','CREB1','CREB3','CREB5',
    'ZBTB16','ZBTB7A','ZBTB7B','BCOR','BCORL1','GFI1','GFI1B',
    'MITF','TFEB','TFE3','TFEC','USF1','USF2',
    'CSRNP1','CSRNP3','E2F1','E2F2','E2F3','E2F4','E2F5','E2F6','E2F7','E2F8',
    'HMGA1','HMGA2','SALL1','SALL4','WT1','EOMES','HHEX','LHX2','LDB1',
    'CUX1','GLIS2','EHMT1','KMT2A','KMT2D','PHF6','ASXL1','ASXL2','SUZ12','EZH2','JARID2',
}

_HEMOGLOBIN_GENES = {'HBA1','HBA2','HBB','HBD','HBE1','HBG1','HBG2','HBZ','HBQ1','HBM'}
_RIBOSOMAL_PREFIXES = ('RPS','RPL','RPP','MRPS','MRPL')
_IG_PREFIXES = ('IGK','IGL','IGH','IGJ','IGLL','IGLC','IGKC','IGHA','IGHG','IGHM','IGHD','IGHE')
_MITO_PREFIXES = ('MT-','MTRNR','ATP6','ATP8','COX1','COX2','COX3','CYTB','ND1','ND2','ND3','ND4','ND5','ND6')

def classify_gene_category(gene):
    """Classify genes as: tf / hemoglobin / ribosomal / immunoglobulin / mitochondrial / other"""
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

# Known key hematopoietic-lineage TFs — additional weighting
_HEMATOPOIETIC_TFS = {
    'SPI1','PAX5','GATA1','GATA2','GATA3','CEBPA','CEBPB','CEBPE',
    'RUNX1','IRF8','EBF1','TCF3','TCF4','TCF12','BCL11A','BCL11B',
    'TAL1','LYL1','LMO2','GFI1','GFI1B','MYB','MYC','ERG','FLI1',
    'NFE2','KLF1','MEIS1','HOXA9','PBX1','ZBTB16','STAT5A','STAT5B',
    'IRF4','PRDM1','BCL6','FOXP3','TBX21','RORC',
}


class DriverGeneScorer:
    """Integrate five evidence dimensions into a unified driver-gene score with prior knowledge of gene functional categories"""

    def __init__(self, evidence, weights=None):
        self.evidence = evidence
        self.weights = weights or {
            'geneWeight': 0.40,
            'DREM':       0.25,
            'dynamic':    0.20,
            'network':    0.10,
            'HC':         0.05,
        }

    def _category_multiplier(self, gene):
        """
        Multiplicative adjustment factors based on gene functional category:
        - TF: 1.30  (boost because it may act as a regulator)
        - hematopoietic TF: additional 1.20 (known lineage regulator)
        - hemoglobin: 0.45 (strong penalty because this is more likely a consequence of anemia than a cause)
        - ribosomal: 0.55 (penalty because these are downstream readouts of translational stress)
        - immunoglobulin: 0.55 (penalty because these are B-cell products)
        - mitochondrial: 0.70 (penalty for mitochondrial genes)
        - other: 1.00
        """
        cat = classify_gene_category(gene)
        mult = 1.0
        if cat == 'hemoglobin':
            mult = 0.45
        elif cat == 'ribosomal':
            mult = 0.55
        elif cat == 'immunoglobulin':
            mult = 0.55
        elif cat == 'mitochondrial':
            mult = 0.70
        elif cat == 'tf':
            mult = 1.30
            if gene.upper() in _HEMATOPOIETIC_TFS:
                mult *= 1.20  # additional bonus for hematopoietic-lineage TFs
        return mult

    def compute(self):
        norm_gw = normalize_dict(self.evidence.get('geneWeight', {}))
        norm_drem = normalize_dict(self.evidence.get('DREM', {}))
        norm_dyn = normalize_dict(self.evidence.get('dynamic', {}))
        norm_net = normalize_dict(self.evidence.get('network', {}))

        hc_set = self.evidence.get('HC', set())
        mfvs_set = self.evidence.get('MFVS_set', set())

        all_genes = (set(norm_gw.keys()) | set(norm_drem.keys()) |
                      set(norm_dyn.keys()) | set(norm_net.keys()))
        print(f"\nTotal number of candidate genes: {len(all_genes)}")

        results = []
        for gene in all_genes:
            gw = norm_gw.get(gene, 0)
            drem = norm_drem.get(gene, 0)
            dyn = norm_dyn.get(gene, 0)
            net = norm_net.get(gene, 0)

            ev_count = ((1 if gw > 0 else 0) + (1 if drem > 0 else 0) +
                         (1 if dyn > 0 else 0) + (1 if net > 0 else 0))

            driver_score = (
                self.weights['geneWeight'] * gw +
                self.weights['DREM'] * drem +
                self.weights['dynamic'] * dyn +
                self.weights['network'] * net
            )

            # Gene-category adjustment
            cat_mult = self._category_multiplier(gene)
            gene_category = classify_gene_category(gene)
            driver_score *= cat_mult

            # HC weighting
            is_hc = gene in hc_set
            driver_score *= (1.0 + self.weights['HC'] * (1 if is_hc else 0))

            # Multi-evidence bonus
            if ev_count >= 2:
                driver_score *= (1.0 + 0.05 * (ev_count - 1))

            gw_raw = self.evidence.get('geneWeight', {}).get(gene, 0)
            drem_raw = self.evidence.get('DREM', {}).get(gene, 0)
            dyn_info = self.evidence.get('dynamic_detail', {}).get(gene, {})
            drem_traj = self.evidence.get('DREM_trajectory_count', {}).get(gene, 0)

            results.append({
                'gene': gene,
                'driver_score': round(driver_score, 6),
                'evidence_count': ev_count,
                'gene_category': gene_category,
                'category_multiplier': round(cat_mult, 3),
                'geneWeight_norm': round(gw, 4),
                'geneWeight_raw': round(gw_raw, 6),
                'DREM_norm': round(drem, 4),
                'DREM_raw': round(drem_raw, 4),
                'DREM_trajectories': drem_traj,
                'dynamic_norm': round(dyn, 4),
                'dynamic_qval': dyn_info.get('min_qval', 1.0),
                'dynamic_log2fc': round(dyn_info.get('max_log2fc', 0), 4),
                'dynamic_track_count': dyn_info.get('track_count', 0),
                'dynamic_direction': dyn_info.get('direction', 'N/A'),
                'network_norm': round(net, 4),
                'is_HC_marker': is_hc,
                'is_MFVS_driver': gene in mfvs_set,
            })

        df = pd.DataFrame(results)
        df = df.sort_values('driver_score', ascending=False).reset_index(drop=True)
        df['rank'] = range(1, len(df) + 1)
        return df


# ============================================================
# Part 4: Statistical Validation
# ============================================================

def permutation_test(df, n_permutations=500):
    """Permutation test: determine whether the driver_score of top-N genes is significantly higher than random"""
    from scipy.stats import percentileofscore

    scores = df['driver_score'].values
    top_n = min(50, len(scores))
    observed_mean = np.mean(scores[:top_n])

    perm_means = []
    rng = np.random.RandomState(42)
    for _ in range(n_permutations):
        perm_means.append(np.mean(rng.choice(scores, top_n, replace=False)))

    perm_means = np.array(perm_means)
    p_value = 1.0 - percentileofscore(perm_means, observed_mean) / 100.0

    print(f"\nPermutation test (n={n_permutations}):")
    print(f"  Top {top_n} mean score: {observed_mean:.6f}")
    print(f"  Random mean: {perm_means.mean():.6f} ± {perm_means.std():.6f}")
    print(f"  Empirical p-value: {p_value:.6f}")
    return {'top_n': top_n, 'observed_mean': observed_mean,
            'random_mean': perm_means.mean(), 'random_std': perm_means.std(),
            'p_value': p_value}


def generate_report(df, output_dir, perm_result=None):
    """Generate a Markdown statistical report"""
    report_path = os.path.join(output_dir, 'driver_gene_report.md')

    top20 = df.head(20)
    lines = [
        "# Caunagi Disease Driver Gene Identification Report",
        "",
        "## Overview",
        f"- Total candidate genes: {len(df)}",
        f"- Genes with multiple evidence dimensions (≥2): {len(df[df['evidence_count'] >= 2])}",
        f"- Genes with at least three evidence dimensions (≥3): {len(df[df['evidence_count'] >= 3])}",
        f"- HC marker genes: {df['is_HC_marker'].sum()}",
        f"- MFVS driver factors: {df['is_MFVS_driver'].sum()}",
        "",
    ]

    if perm_result:
        lines += [
            "## Statistical Significance",
            f"- Top {perm_result['top_n']} gene mean score: **{perm_result['observed_mean']:.6f}**",
            f"- Random expectation: {perm_result['random_mean']:.6f} ± {perm_result['random_std']:.6f}",
            f"- Empirical p-value: **{perm_result['p_value']:.6f}**",
            "",
        ]

    lines += [
        "## Top 20 Candidate Driver Genes",
        "",
        "| Rank | Gene | Score | #Evid | GeneW | DREM | Dyn | Net | HC | MFVS |",
        "|------|------|-------|-------|-------|------|-----|-----|-----|------|",
    ]

    for _, row in top20.iterrows():
        lines.append(
            f"| {int(row['rank'])} | {row['gene']} | {row['driver_score']:.4f} | "
            f"{int(row['evidence_count'])} | {row['geneWeight_norm']:.3f} | "
            f"{row['DREM_norm']:.3f} | {row['dynamic_norm']:.3f} | "
            f"{row['network_norm']:.3f} | "
            f"{'✓' if row['is_HC_marker'] else ''} | "
            f"{'✓' if row['is_MFVS_driver'] else ''} |"
        )

    lines += [
        "",
        "## Top 10 Genes for Each Evidence Dimension",
    ]

    for dim, label in [('geneWeight_raw', 'GeneWeight'), ('DREM_raw', 'DREM'), 
                         ('dynamic_norm', 'Dynamic Marker')]:
        top = df.nlargest(10, dim)[['gene', dim]].values if dim in df.columns else []
        lines.append(f"\n### {label}")
        lines.append("| Gene | Score |")
        lines.append("|------|-------|")
        for gene, score in top:
            lines.append(f"| {gene} | {score:.4f} |")

    lines += [
        "",
        "## Evidence Dimension Weights",
        "| Evidence | Weight | Description |",
        "|------|------|------|",
        "| geneWeight | 40% | Gene importance learned during iterative Caunagi training |",
        "| DREM TF | 25% | TF regulatory activity inferred by iDREM (−log10(p) × log(number of trajectories)) |",
        "| Dynamic Markers | 20% | Significance of genes showing monotonic changes during disease progression |",
        "| Network Topology | 10% | Centrality position in the regulatory network (Betweenness + PageRank + out-degree + MFVS) |",
        "| HC Markers | 5% | Hierarchical-clustering differential genes (bonus term) |",
        "",
        '> **Note**: The driver genes identified in this report are "disease-progression regulatory factors"—genes that may regulate expression during disease onset or progression—rather than conventional "disease-causing mutated genes.\"',
    ]

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\nReport saved to: {report_path}")


# ============================================================
# Part 5: Main Entry Point
# ============================================================

def main(temp_path, final_iteration, prior_net_path=None,
         output_dir=None, species='human'):
    """
    Main function for Caunagi driver-gene identification

    Parameters:
        temp_path: Caunagi output directory (contains stagedata/, idremResults/, dynamic_markers.pkl, and hcmarkers.pkl)
        final_iteration: final iteration number (e.g., 5)
        prior_net_path: path to the prior regulatory network (e.g., NicheNet_human.csv; network analysis is skipped if not provided)
        output_dir: result output directory (default: temp_path/driver_results/)
        species: 'human' or 'mouse'
    """
    if output_dir is None:
        output_dir = os.path.join(temp_path, 'driver_results')
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 60)
    print("  Caunagi Disease Driver Gene Identification")
    print("  Five-Dimensional Evidence Integration Framework")
    print("=" * 60)
    print(f"  Data directory: {temp_path}")
    print(f"  Final iteration: {final_iteration}")
    print(f"  Output directory: {output_dir}")
    print("=" * 60)

    # Step 1: Collect evidence
    collector = CaunagiEvidenceCollector(temp_path, final_iteration, species)
    collector.collect_all(prior_net_path=prior_net_path)

    # Step 2: Integrate scores
    print("\n" + "=" * 60)
    print("  Multi-Evidence Integrated Scoring")
    print("=" * 60)
    scorer = DriverGeneScorer(collector.evidence)
    df = scorer.compute()

    # Step 3: Permutation test
    perm_result = permutation_test(df, n_permutations=500)

    # Step 4: Output
    csv_path = os.path.join(output_dir, 'driver_gene_candidates.csv')
    df.to_csv(csv_path, index=False)
    print(f"\nComplete results saved to: {csv_path}")

    generate_report(df, output_dir, perm_result)

    # Print Top 25 to terminal
    print("\n" + "=" * 80)
    print("TOP 25 Candidate Driver Genes")
    print("=" * 80)
    header = f"{'Rank':<5} {'Gene':<12} {'Score':<9} {'#Ev':<4} {'Cat':<10} {'GeneW':<7} {'DREM':<7} {'Dyn':<7} {'Net':<7} {'MFVS':<5}"
    print(header)
    print("-" * 80)
    for _, r in df.head(25).iterrows():
        print(f"{int(r['rank']):<5} {r['gene']:<12} {r['driver_score']:.4f}   "
              f"{int(r['evidence_count']):<4} {r['gene_category']:<10} "
              f"{r['geneWeight_norm']:.4f}  "
              f"{r['DREM_norm']:.4f}  {r['dynamic_norm']:.4f}  {r['network_norm']:.4f}  "
              f"{'Y' if r['is_MFVS_driver'] else 'N':<5}")

    print("\n" + "=" * 60)
    print("  Top 15 Detailed Evidence")
    print("=" * 60)
    for i, (_, r) in enumerate(df.head(15).iterrows()):
        print(f"\n{i+1}. {r['gene']}  (driver_score={r['driver_score']:.4f})")
        if r['geneWeight_raw'] > 0:
            print(f"   GeneWeight: {r['geneWeight_raw']:.6f}")
        if r['DREM_raw'] > 0:
            print(f"   DREM: {r['DREM_raw']:.4f} (appears in {r['DREM_trajectories']} trajectories)")
        if r['dynamic_norm'] > 0:
            print(f"   Dynamic: q={r['dynamic_qval']:.2e}, |log2FC|={r['dynamic_log2fc']:.4f}, "
                  f"trajectory_count={r['dynamic_track_count']}, direction={r['dynamic_direction']}")
        if r['network_norm'] > 0:
            print(f"   Network: centrality={r['network_norm']:.4f}")
        if r['is_HC_marker']:
            print(f"   HC Marker: YES")
        if r['is_MFVS_driver']:
            print(f"   MFVS Driver: YES")

    print("\nDone!")
    return df


if __name__ == '__main__':
    # ============ User Configuration ============
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
    TEMP_PATH = os.path.join(PROJECT_ROOT, "temp_path")
    FINAL_ITERATION = 5
    PRIOR_NET_PATH = os.path.join(TEMP_PATH, "NicheNet_human.csv")
    OUTPUT_DIR = SCRIPT_DIR
    # ===================================

    df = main(
        TEMP_PATH,
        FINAL_ITERATION,
        prior_net_path=PRIOR_NET_PATH,
        output_dir=OUTPUT_DIR,
    )

