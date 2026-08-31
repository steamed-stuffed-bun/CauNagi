# CauNagi
CauNagi is a causal representation learning and temporal regulatory analysis framework for multi-stage single-cell transcriptomic data. The core workflow combines causal concept disentanglement, Gaussian diffusion, stage-wise clustering, temporal graph construction, iDREM analysis, iterative `geneWeight` updates, and static/dynamic marker discovery.

This repository contains the complete contents of the project's `Main` folder. Perturbation simulation, drug-response simulation, CMAP analysis, and other unrelated experimental code are not part of the core pipeline.

## Core modules

| Module | Purpose |
|---|---|
| `caunagi_main.py` | Public `Caunagi` API |
| `trainer.py` / `Module.py` | Diffusion model and training loop |
| `runner.py` | One complete CauNagi iteration |
| `utils.py` | Data loading and shared utilities |
| `CPO_utils.py` | Clustering parameter optimization |
| `buildGraph.py` / `distDistance.py` | Temporal graph construction and cluster distances |
| `processIDREM.py` / `processTFs.py` | iDREM execution and TF/target-gene parsing |
| `attribute_utils.py` | AnnData attributes and dataset merging |
| `dynamic_markers.py` | Dynamic progression marker discovery |
| `hierachical_static_markers.py` | Hierarchical static marker discovery |
| `get_driver.py` | Post-hoc marker analysis entry point |

## Installation

Use Python 3.10 or newer and create an isolated environment:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install numpy pandas anndata scanpy scipy scikit-learn einops tqdm joblib h5py networkx leidenalg igraph umap-learn
```

Install PyTorch separately according to the available CPU/GPU and CUDA version. Verify the installation with:

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

The full training workflow is designed for a CUDA-capable GPU, but the API can use CPU when CUDA is unavailable. Java is also required for iDREM:

```powershell
java -version
```

## Input data

The input directory passed to `process_data()` must contain one AnnData file per stage:

```text
input_data/
├── 0.h5ad
├── 1.h5ad
└── ...
```

Every file must contain an expression matrix in `adata.X`, consistent `adata.var_names`, and a `CellType` column in `adata.obs`. Columns named in `concept_list` must also exist in `adata.obs`.

The causal concept matrix must have one extra row and column for the unexplained concept:

```python
concept_list = ["disease_stage", "CellType"]
concept_cdag = np.zeros(
    (len(concept_list) + 1, len(concept_list) + 1),
    dtype=np.float32,
)
```

## iDREM requirements

`run_caunagi()` requires an iDREM directory containing `idrem.jar`, `example_settings.txt`, and the required Human or Mouse reference files, including TF-gene interactions and gene annotations. Java must be available on `PATH`.

## Minimal usage

Run the following from the directory that contains the cloned repository. Replace the paths and concept graph with the settings for your experiment.

```python
from pathlib import Path
import numpy as np

from CauNagi import Caunagi

input_dir = Path("input_data")
temp_dir = Path("caunagi_run")
model_dir = Path("model_checkpoints")
idrem_dir = Path("idrem")

concept_list = ["disease_stage", "CellType"]
concept_cdag = np.zeros(
    (len(concept_list) + 1, len(concept_list) + 1),
    dtype=np.float32,
)

model = Caunagi(
    concept_list=concept_list,
    concept_cdag=concept_cdag,
    total_stage_num=2,
    device="cuda",
)

model.process_data(
    data_path=input_dir,
    temp_path=temp_dir,
    stage_name="disease_stage",
    iteration=0,
    celltype_concept_name="CellType",
    disease_idx={"healthy": 0, "disease": 1},
    log_norm=True,
)

model.setup_train(
    model_save_path=model_dir,
    epoch_num=100000,
    training_batch_size=64,
    training_lr=2e-5,
    max_profile_size=2000,
    timesteps=1000,
)
model.register_species("Human")
model.register_CPO_parameters(
    anchor_neighbors=15,
    max_neighbors=35,
    min_neighbors=10,
    resolution_min=0.8,
    resolution_max=1.5,
)
model.register_iDREM_parameters(
    Minimum_Absolute_Log_Ratio_Expression=0.5,
    Convergence_Likelihood=0.001,
    Minimum_Standard_Deviation=0.5,
)

model.run_caunagi(idrem_dir=idrem_dir, CPO=True)
model.analyse_UNAGI(
    data_path=temp_dir / "0" / "stagedata",
    iteration=0,
    progressionmarker_background_sampling_times=1000,
    save_dir=Path("results") / "caunagi_markers",
)
```

The `data_path` passed to `analyse_UNAGI()` must be the final iteration's `stagedata` directory. The model checkpoint directory must be the same directory supplied to `setup_train()`.

## Iteration outputs

Each iteration writes staged data, iDREM inputs/results, graph edges, representations, and merged AnnData outputs beneath `temp_dir/<iteration>/`. Downstream result tables and figures should be written to a separate `results/` directory. Large local data and model files are excluded by `.gitignore`.

## Notes

- Set `CUDA_VISIBLE_DEVICES` before importing the package when selecting a specific GPU.
- The two distance modules are intentionally kept synchronized because different parts of the pipeline import each one.
- A later iteration consumes the previous iteration's staged data and `geneWeight` layer.
- Do not commit passwords, access tokens, private datasets, checkpoints, or iDREM reference archives.
