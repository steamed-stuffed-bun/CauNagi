"""Post-hoc analyses that are part of the CauNagi workflow.

The core downstream analyses described by CauNagi are hierarchical static
markers and progression markers. Perturbation, drug-response, and pathway
simulation code is intentionally not part of this module because those
modules are not required by the CauNagi method.
"""

from pathlib import Path
import pickle

import numpy as np
import scanpy as sc

from .dynamic_markers import runGetProgressionMarker_one_dist
from .dynamic_markers_helper import get_progressionmarker_background
from .hierachical_static_markers import get_dataset_hcmarkers


class Analyst:
    """Run marker analyses on a completed CauNagi iteration."""

    def __init__(self, data_path, iteration, target_dir=None):
        input_path = Path(data_path)
        self.dataset_path = (
            input_path / "dataset.h5ad" if input_path.is_dir() else input_path
        )
        if not self.dataset_path.is_file():
            raise FileNotFoundError(
                f"The staged dataset was not found: {self.dataset_path}"
            )

        self.data_dir = self.dataset_path.parent
        attribute_path = self.data_dir / "attribute.pkl"
        if not attribute_path.is_file():
            raise FileNotFoundError(
                f"The staged dataset must be accompanied by {attribute_path}."
            )

        self.adata = sc.read_h5ad(self.dataset_path)
        with attribute_path.open("rb") as handle:
            self.adata.uns = pickle.load(handle)

        required_obs = {"stage", "leiden"}
        missing_obs = required_obs.difference(self.adata.obs.columns)
        if missing_obs:
            raise ValueError(
                "The staged dataset is missing required observations: "
                + ", ".join(sorted(missing_obs))
            )

        self.total_stage = self.adata.obs["stage"].nunique()
        if self.total_stage < 2:
            raise ValueError("At least two stages are required for marker analysis.")

        self.iteration = iteration
        self.target_dir = Path(target_dir) if target_dir else self.data_dir
        self.target_dir.mkdir(parents=True, exist_ok=True)

    def start_analyse(
        self,
        progressionmarker_background_sampling=1000,
        *,
        ignore_dynamic_markers=False,
        ignore_hcmarkers=False,
    ):
        """Run the non-perturbational CauNagi downstream analyses.

        Results are written to ``target_dir`` as ``hcmarkers.pkl``,
        ``dynamic_markers.pkl``, and a cached progression-marker background.
        """
        if progressionmarker_background_sampling < 1:
            raise ValueError("Background sampling must be a positive integer.")

        if not ignore_hcmarkers:
            representation = "X_umap" if "X_umap" in self.adata.obsm else "z"
            hcmarkers = get_dataset_hcmarkers(
                self.adata,
                stage_key="stage",
                cluster_key="leiden",
                use_rep=representation,
            )
            self.adata.uns["hcmarkers"] = hcmarkers
            with (self.target_dir / "hcmarkers.pkl").open("wb") as handle:
                pickle.dump(hcmarkers, handle)

        if not ignore_dynamic_markers:
            idrem_dir = self.data_dir.parent / "idremResults"
            if not idrem_dir.is_dir():
                raise FileNotFoundError(
                    f"The iDREM results directory was not found: {idrem_dir}"
                )

            background_path = (
                self.target_dir
                / f"{progressionmarker_background_sampling}_progressionmarker_background.npy"
            )
            if background_path.is_file():
                background = np.load(background_path, allow_pickle=True).item()
            else:
                background = get_progressionmarker_background(
                    times=progressionmarker_background_sampling,
                    adata=self.adata,
                    total_stage=self.total_stage,
                )
                np.save(background_path, background, allow_pickle=True)

            dynamic_markers = runGetProgressionMarker_one_dist(
                str(idrem_dir),
                background,
                self.adata.shape[1],
                cutoff=0.05,
            )
            self.adata.uns["progressionMarkers"] = dynamic_markers
            with (self.target_dir / "dynamic_markers.pkl").open("wb") as handle:
                pickle.dump(dynamic_markers, handle)

        return self.adata


# Preserve the original public name for existing user scripts.
analyst = Analyst
