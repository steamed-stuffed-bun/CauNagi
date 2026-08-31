"""
Figure 3 — Ablation Experiment Bar Chart
Caunagi Disentanglement Benchmark

背景:
    - 消融对象是 Caunagi 的三个变体，每个变体是一个 *独立的代码文件夹*（各自一份
      完整的 caunagi 包），而不是通过给原版加参数得到的：
          Caunagi                  -> Full Caunagi        （原版）
          w_o_Disentanglement      -> w/o Disentanglement （切除监督解耦: 标签预测+对抗）
          w_o_Causal_DAG           -> w/o Causal DAG      （切除概念间因果有向结构）
          w_o_Iterative_Feedback   -> w/o Iter. Feedback  （切除迭代基因权重自聚焦机制）
      四个文件夹都放在本脚本所在的 Ablation/ 目录下。
    - 消融实验的输入数据统一放在:
          /disk1/cai029/biosoft/UNAGI/UNAGI/data/example/

用法:
    python plot_figure3.py
    输出: figure3_ablation.pdf + figure3_ablation.png
          ablation_metrics.csv  （运行得到的原始指标缓存）
"""

import os
import sys
import gc
import shutil
import importlib

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# 0. 全局样式
# ============================================================
matplotlib.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.05,
})

# ============================================================
# 1. 实验配置 —— 已按真实路径 / 目录结构填好
# ============================================================
# 本脚本所在目录，即 Ablation/ （四个变体文件夹都在这里）
ABLATION_ROOT = os.path.dirname(os.path.abspath(__file__))

# 消融实验统一输入数据路径
DATA_PATH = "/disk1/cai029/biosoft/UNAGI/UNAGI/data/example/"

# iDREM 可执行目录（与 Ablation.ipynb 保持一致）
IDREM_PATH = "/disk1/cai029/biosoft/idrem"

# 5 个随机种子 -> 5 次独立重复实验（与 Ablation.ipynb 一致）
SEEDS = [888, 999, 101, 202, 303]

# 迭代次数（迭代基因权重自聚焦机制依赖多轮迭代）
ITER_NUM = 5

# 变体展示名  ->  Ablation/ 下对应的独立包文件夹名
# 注意: 每个变体是单独的一份代码，靠“换文件夹 import”来切换，而非加参数
VARIANT_DIRS = {
    "Full Caunagi":        "Caunagi",
    "w/o Disentanglement": "w_o_Disentanglement",
    "w/o Causal DAG":      "w_o_Causal_DAG",
    "w/o Iter. Feedback":  "w_o_Iterative_Feedback",
}

# 模型超参（与 Ablation.ipynb 一致）
CONCEPT_LIST = ["stage", "name.simple"]
CONCEPT_CDAG = [[0, 0, 0], [1, 0, 0], [0, 0, 0]]
STATE_IDX = {0: 0, 1: 1, 2: 2, 3: 3}
TOTAL_STAGE_NUM = 4
STAGE_NAME = "stage"
CELLTYPE_CONCEPT_NAME = "name.simple"

# 计算 SCIB 指标时需要的字段（按你数据里的实际字段名调整）
EMBED_KEY = "X_emb"              # adata.obsm 中的表征键
LABEL_KEY = "name.simple"       # 细胞类型标签列（真值）

# 运行结果缓存：跑一次后写到这里，之后直接读，避免重复重训
CACHE_CSV = os.path.join(ABLATION_ROOT, "ablation_metrics.csv")

METRICS = [
    "ARI", "NMI", "Label Score", "Silhouette",
    "cLISI graph", "Overall SCIB",
]


# ============================================================
# 1a. 动态加载某个变体的独立代码包
# ============================================================
def _load_variant_module(folder):
    """从 Ablation/<folder>/ 载入该变体自己的 caunagi_main 模块。

    每个变体是一份完整独立的包（含相对 import），因此切换变体时必须先把上一个
    变体缓存的模块清掉，再重新 import，保证 `from .Module import ...` 绑定到
    当前变体文件夹下的实现。
    """
    if ABLATION_ROOT not in sys.path:
        sys.path.insert(0, ABLATION_ROOT)

    # 清理上一个变体残留的模块缓存
    for mod_name in list(sys.modules):
        if mod_name == folder or mod_name.startswith(folder + "."):
            del sys.modules[mod_name]

    return importlib.import_module(f"{folder}.caunagi_main")


# ============================================================
# 1b. 跑单个 (变体, seed)，返回最终迭代得到的 AnnData（含表征）
# ============================================================
def _run_variant_seed(folder, seed):
    import scanpy as sc

    caunagi_main = _load_variant_module(folder)

    # 每个变体 + seed 用独立的临时目录 / 模型目录，互不干扰
    tag = f"{folder}_seed{seed}"
    temp_path = os.path.join(DATA_PATH, f"temp_path_{tag}")
    model_save = os.path.join(DATA_PATH, f"model_save_{tag}") + "/"

    # process_data 在 iteration==0 时要求 temp_path 不存在
    if os.path.exists(temp_path):
        shutil.rmtree(temp_path)
    os.makedirs(model_save, exist_ok=True)

    caunagi = caunagi_main.Caunagi(
        CONCEPT_LIST,
        CONCEPT_CDAG,
        total_stage_num=TOTAL_STAGE_NUM,
        save_and_sample_every=1000,
    )

    for iteration in range(ITER_NUM):
        # 第 0 轮读原始输入；之后读上一轮迭代产出的 stagedata
        train_data_path = DATA_PATH if iteration == 0 else temp_path
        caunagi.process_data(
            data_path=train_data_path,
            temp_path=temp_path,
            stage_name=STAGE_NAME,
            iteration=iteration,
            celltype_concept_name=CELLTYPE_CONCEPT_NAME,
            disease_idx=STATE_IDX,
        )
        caunagi.setup_train(
            model_save_path=model_save,
            epoch_num=2000,
            max_profile_size=3000,
            seed=seed,
        )
        caunagi.run_caunagi(IDREM_PATH)

    # 最后一轮迭代的表征结果
    final_h5ad = os.path.join(temp_path, f"{ITER_NUM - 1}/stagedata/dataset.h5ad")
    adata = sc.read_h5ad(final_h5ad)

    del caunagi
    gc.collect()
    try:
        import torch
        torch.cuda.empty_cache()
    except Exception:
        pass

    return adata


# ============================================================
# 1c. 从一份表征 AnnData 计算 6 个指标
# ============================================================
def _compute_metrics(adata):
    """在表征空间上计算 ARI / NMI / Label Score / Silhouette / cLISI graph / Overall SCIB。

    - EMBED_KEY / LABEL_KEY 见上方配置。
    - Label Score 用表征上的 kNN 标签迁移准确率。
    - Overall SCIB 取其余 5 个指标的均值（本图内自洽的综合分）。
    """
    import scanpy as sc
    from sklearn.metrics import (
        adjusted_rand_score,
        normalized_mutual_info_score,
        silhouette_score,
    )
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.model_selection import cross_val_score

    emb = np.asarray(adata.obsm[EMBED_KEY])
    labels = adata.obs[LABEL_KEY].astype(str).values

    # 在表征上做 leiden 聚类，用于 ARI / NMI
    sc.pp.neighbors(adata, use_rep=EMBED_KEY)
    sc.tl.leiden(adata, key_added="_abl_cluster")
    clusters = adata.obs["_abl_cluster"].astype(str).values

    ari = adjusted_rand_score(labels, clusters)
    nmi = normalized_mutual_info_score(labels, clusters)

    # kNN 标签迁移准确率（5 折）
    knn = KNeighborsClassifier(n_neighbors=15)
    label_score = float(np.mean(cross_val_score(knn, emb, labels, cv=5)))

    # Silhouette（按细胞类型），映射到 [0,1]
    sil_raw = silhouette_score(emb, labels)
    silhouette = (sil_raw + 1.0) / 2.0

    # cLISI graph（细胞类型局部一致性），需要 scib
    try:
        import scib
        clisi = float(scib.metrics.clisi_graph(
            adata, label_key=LABEL_KEY, type_="embed", use_rep=EMBED_KEY
        ))
    except Exception as e:
        print(f"    [warn] cLISI graph 计算失败，置为 NaN: {e}")
        clisi = np.nan

    parts = [ari, nmi, label_score, silhouette, clisi]
    overall = float(np.nanmean(parts))

    return {
        "ARI": ari,
        "NMI": nmi,
        "Label Score": label_score,
        "Silhouette": silhouette,
        "cLISI graph": clisi,
        "Overall SCIB": overall,
    }


# ============================================================
# 1d. 主入口：跑全部变体 × 全部 seed，产出长表 DataFrame
# ============================================================
def load_real_data(use_cache=True):
    """运行四个变体（各自独立文件夹）× 5 个 seed，计算指标并返回长表。

    返回列: variant, metric, value, run, seed
    结果会缓存到 CACHE_CSV，二次运行直接读缓存。
    """
    if use_cache and os.path.exists(CACHE_CSV):
        print(f"[cache] 读取已有结果: {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    records = []
    for variant, folder in VARIANT_DIRS.items():
        variant_dir = os.path.join(ABLATION_ROOT, folder)
        if not os.path.isdir(variant_dir):
            print(f"[skip] 找不到变体文件夹: {variant_dir}")
            continue

        for run_idx, seed in enumerate(SEEDS, start=1):
            print(f"[run] {variant} ({folder}) | seed={seed} | run={run_idx}")
            adata = _run_variant_seed(folder, seed)
            metrics = _compute_metrics(adata)
            for metric, value in metrics.items():
                records.append({
                    "variant": variant,
                    "metric": metric,
                    "value": value,
                    "run": run_idx,
                    "seed": seed,
                })

    df = pd.DataFrame(records)
    df.to_csv(CACHE_CSV, index=False)
    print(f"[cache] 结果已写入: {CACHE_CSV}")
    return df


# ---- 加载真实消融结果 ----
df = load_real_data()


# ============================================================
# 2. 数据验证
# ============================================================
REQUIRED_COLS = {"variant", "metric", "value", "run"}
assert REQUIRED_COLS.issubset(df.columns), \
    f"DataFrame 缺少必需列: {REQUIRED_COLS - set(df.columns)}"

EXPECTED_VARIANTS = set(VARIANT_DIRS.keys())
EXPECTED_METRICS = set(METRICS)

actual_variants = set(df["variant"].unique())
actual_metrics = set(df["metric"].unique())

if actual_variants != EXPECTED_VARIANTS:
    print(f"WARNING: 变体不匹配。期望 {EXPECTED_VARIANTS}，实际 {actual_variants}")
if actual_metrics != EXPECTED_METRICS:
    print(f"WARNING: 指标不匹配。期望 {EXPECTED_METRICS}，实际 {actual_metrics}")

n_runs_per_group = df.groupby(["variant", "metric"]).size()
if (n_runs_per_group != len(SEEDS)).any():
    print(f"WARNING: 部分 (variant, metric) 组合的 run 数不为 {len(SEEDS)}:")
    print(n_runs_per_group[n_runs_per_group != len(SEEDS)])


# ============================================================
# 3. 计算统计量
# ============================================================
summary = (
    df.groupby(["variant", "metric"])["value"]
    .agg(mean="mean", std="std", count="count")
    .reset_index()
)
summary["sem"] = summary["std"] / np.sqrt(summary["count"])

print("\n=== 汇总统计 ===")
print(summary.pivot(index="variant", columns="metric", values="mean").round(4))


# ============================================================
# 4. 显著性检验（配对 t 检验）
# ============================================================
VARIANTS = list(VARIANT_DIRS.keys())  # Full → w/o Dis → w/o DAG → w/o Iter
COLORS = ["#2166AC", "#92C5DE", "#F4A582", "#CA0020"]
ABLATION_VARIANTS = VARIANTS[1:]  # 三个消融变体

significance = {}  # {(metric, ablation_variant): p_value}
pivot_full = df[df["variant"] == "Full Caunagi"].pivot(
    index="run", columns="metric", values="value"
)

for variant_abl in ABLATION_VARIANTS:
    pivot_abl = df[df["variant"] == variant_abl].pivot(
        index="run", columns="metric", values="value"
    )
    for metric in METRICS:
        # 对齐 run 以保证配对正确
        common_runs = pivot_full.index.intersection(pivot_abl.index)
        if len(common_runs) < 3:
            significance[(metric, variant_abl)] = 1.0  # 样本不足，不显著
            continue
        full_vals = pivot_full.loc[common_runs, metric].values
        abl_vals = pivot_abl.loc[common_runs, metric].values
        t_stat, p_val = stats.ttest_rel(full_vals, abl_vals)
        significance[(metric, variant_abl)] = p_val


# ============================================================
# 5. 绘图
# ============================================================
def p_to_stars(p):
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    return None


fig, ax = plt.subplots(figsize=(13, 5.2))

x = np.arange(len(METRICS))
n_variants = len(VARIANTS)
total_width = 0.80
bar_width = total_width / n_variants

for v_idx, (variant, color) in enumerate(zip(VARIANTS, COLORS)):
    means = []
    sems = []
    for metric in METRICS:
        row = summary[(summary["variant"] == variant) & (summary["metric"] == metric)]
        if row.empty:
            means.append(0)
            sems.append(0)
        else:
            means.append(row["mean"].values[0])
            sems.append(row["sem"].values[0])

    offset = (v_idx - (n_variants - 1) / 2) * bar_width

    ax.bar(
        x + offset,
        means,
        bar_width,
        yerr=sems,
        label=variant,
        color=color,
        edgecolor="white",
        linewidth=0.3,
        capsize=2.5,
        error_kw={"linewidth": 1.0, "color": "#555555"},
        zorder=3,
    )

    # 显著性星号（仅消融变体）
    if variant != "Full Caunagi":
        for m_idx, metric in enumerate(METRICS):
            p = significance.get((metric, variant), 1.0)
            stars = p_to_stars(p)
            if stars:
                bar_top = means[m_idx] + sems[m_idx]
                ax.text(
                    x[m_idx] + offset,
                    bar_top + 0.012,
                    stars,
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    fontweight="bold",
                    color=color,
                )

# ---- 坐标轴 & 标注 ----
ax.set_xticks(x)
ax.set_xticklabels(METRICS, fontsize=11)
ax.set_ylabel("Score", fontsize=12, fontweight="medium")
ax.set_ylim(0, 1.08)

# 图例
handles, labels = ax.get_legend_handles_labels()
order = [0, 1, 2, 3]  # Full → w/o Dis → w/o DAG → w/o Iter
ax.legend(
    [handles[i] for i in order],
    [labels[i] for i in order],
    loc="upper right",
    fontsize=9,
    ncol=2,
    frameon=True,
    edgecolor="#cccccc",
    framealpha=1.0,
)

# 去边框
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_linewidth(0.8)
ax.spines["bottom"].set_linewidth(0.8)

# 网格
ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.5)
ax.set_axisbelow(True)

# 底部注释
fig.text(
    0.5, -0.02,
    f"n = {len(SEEDS)} independent runs per variant; stars indicate paired t-test vs Full Caunagi: "
    "*p < 0.05, **p < 0.01, ***p < 0.001",
    ha="center", fontsize=7.5, color="#666666",
)

plt.tight_layout(rect=[0, 0.02, 1, 1])
plt.savefig("figure3_ablation.pdf", dpi=300)
plt.savefig("figure3_ablation.png", dpi=300)
plt.show()

print("\n=== 显著性矩阵 (p-values, 未校正) ===")
sig_df = pd.DataFrame(index=METRICS, columns=ABLATION_VARIANTS)
for m in METRICS:
    for v in ABLATION_VARIANTS:
        sig_df.loc[m, v] = f"{significance.get((m, v), 1.0):.4f}"
print(sig_df)
