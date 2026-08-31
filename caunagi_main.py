"""Public CauNagi API.

The public workflow is:

1. preprocess stage-specific AnnData files with :meth:`process_data`;
2. create the diffusion model with :meth:`setup_train`;
3. optionally register CPO, species, and iDREM settings;
4. run one or more CauNagi iterations with :meth:`run_caunagi`;
5. run marker analyses with :meth:`analyse_UNAGI`.
"""

import os

# Respect a CUDA choice made by the user before importing torch.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import random
from copy import deepcopy
from pathlib import Path

import anndata as ad
import joblib
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.sparse import issparse

from .Module import Denoise_net, GaussianDiffusion
from .get_driver import Analyst
from .runner import Caunagi_runner
from .utils import merge_data
from .trainer import Trainer


class Caunagi:
    """Main CauNagi model and iterative analysis interface."""

    def __init__(
        self,
        concept_list,
        concept_cdag,
        total_stage_num,
        device="cuda",
        ema_decay=0.995,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=1000,
        save_and_sample_every=100,
    ):
        if total_stage_num < 2:
            raise ValueError("CauNagi requires at least two stages.")
        if len(concept_cdag) != len(concept_list) + 1:
            raise ValueError(
                "The causal DAG must include one additional unexplained concept."
            )

        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            print("CUDA is unavailable; falling back to CPU.")
            requested_device = torch.device("cpu")

        self.device = requested_device
        self.ema_decay = ema_decay
        self.gradient_accumulate_every = gradient_accumulate_every
        self.fp16 = fp16
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.save_and_sample_every = save_and_sample_every
        self.model = None
        self.trainer = None
        self.CPO_parameters = None
        self.iDREM_parameters = None
        self.species = "Human"
        self.concept_list = list(concept_list)
        self.concept_cdag = concept_cdag
        self.total_stage_num = total_stage_num
        self.input_data = None
        self.CellType_dicts = None
        self.iteration = None

    def process_data(
        self,
        data_path,
        temp_path,
        stage_name,
        iteration,
        celltype_concept_name,
        disease_idx,
        log_norm=False,
    ):
        """Prepare stage data and encode categorical concepts as integers.

        On iteration zero, ``data_path`` must contain ``0.h5ad`` through
        ``N.h5ad``. Later iterations read the previous iteration's staged
        dataset from ``temp_path``.
        """
        data_dir = Path(data_path)
        temp_dir = Path(temp_path)
        if iteration == 0:
            if not data_dir.is_dir():
                raise FileNotFoundError(f"Input data directory not found: {data_dir}")
            if temp_dir.exists():
                raise ValueError(
                    f"The temporary directory already exists: {temp_dir}. "
                    "Choose a new path or remove the old run explicitly."
                )
            temp_dir.mkdir(parents=True)
            input_data = merge_data(data_dir, self.total_stage_num)
        else:
            previous_dataset = (
                temp_dir / str(iteration - 1) / "stagedata" / "dataset.h5ad"
            )
            if not previous_dataset.is_file():
                raise FileNotFoundError(
                    f"Previous iteration dataset not found: {previous_dataset}"
                )
            input_data = sc.read_h5ad(previous_dataset)

        if stage_name not in input_data.obs:
            raise KeyError(f"Stage column not found in input data: {stage_name}")
        if celltype_concept_name not in input_data.obs:
            raise KeyError(
                f"Cell-type concept column not found in input data: "
                f"{celltype_concept_name}"
            )
        if not set(self.concept_list).issubset(input_data.obs.columns):
            missing = set(self.concept_list).difference(input_data.obs.columns)
            raise KeyError("Missing concept columns: " + ", ".join(sorted(missing)))
        if stage_name not in self.concept_list:
            raise ValueError("stage_name must be one of concept_list.")

        expression = input_data.X.toarray() if issparse(input_data.X) else np.asarray(input_data.X)
        expression = expression.astype(np.float32, copy=False)
        if log_norm:
            row_sums = np.maximum(expression.sum(axis=1, keepdims=True), 1.0)
            expression = np.log1p(expression / row_sums * 10000).astype(np.float32)

        encoded_obs = []
        label_categories = []
        for factor_name in self.concept_list:
            if factor_name == stage_name:
                mapping = dict(disease_idx)
            else:
                values = list(pd.unique(input_data.obs[factor_name]))
                mapping = {value: index for index, value in enumerate(values)}

            unknown = set(input_data.obs[factor_name]).difference(mapping)
            if unknown:
                raise ValueError(
                    f"Values in {factor_name} are missing from its mapping: {unknown}"
                )
            encoded_obs.append(input_data.obs[factor_name].map(mapping).to_numpy())
            label_categories.append(len(mapping))
            joblib.dump(mapping, data_dir / f"{factor_name}_dict.pkl")
            if factor_name == celltype_concept_name:
                self.CellType_dicts = mapping

        concept_obs = pd.DataFrame(
            np.column_stack(encoded_obs), columns=self.concept_list, index=input_data.obs_names
        )
        processed = ad.AnnData(
            X=expression,
            obs=concept_obs,
            var=input_data.var.copy(),
        )
        processed.write_h5ad(data_dir / "process_data.h5ad")

        self.data_path = data_dir
        self.temp_path = temp_dir
        self.stage_name = stage_name
        self.iteration = iteration
        self.ProcessData_path = data_dir / "process_data.h5ad"
        self.input_data = input_data
        self.concept_counts = label_categories
        print("The input dataset has been processed.")

    def setup_train(
        self,
        model_save_path,
        *,
        loss_type="l2",
        epoch_num=100000,
        training_batch_size=64,
        training_lr=2e-5,
        max_profile_size=2000,
        timesteps=1000,
        seed=888,
        train_log=True,
    ):
        """Build the diffusion model and its trainer."""
        if not hasattr(self, "ProcessData_path"):
            raise RuntimeError("Call process_data before setup_train.")

        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

        train_data = sc.read_h5ad(self.ProcessData_path)
        training_profile_size = min(max_profile_size, train_data.n_vars)
        self.training_profile_size = training_profile_size
        self.epoch_num = epoch_num
        self.output_path = Path(model_save_path)
        self.output_path.mkdir(parents=True, exist_ok=True)

        denoise_net = Denoise_net(
            training_profile_size,
            training_profile_size,
            len(self.concept_counts) + 1,
            torch.as_tensor(self.concept_cdag, dtype=torch.float32),
            self.concept_counts,
        ).to(self.device)
        diffusion_model = GaussianDiffusion(
            denoise_net,
            profile_size=training_profile_size,
            timesteps=timesteps,
            loss_type=loss_type,
        ).to(self.device)

        gene_weights = None
        if self.iteration and self.input_data is not None:
            if "geneWeight" in self.input_data.layers:
                layer = self.input_data.layers["geneWeight"]
                gene_weights = layer.toarray() if hasattr(layer, "toarray") else np.asarray(layer)
                gene_weights = gene_weights[:, :training_profile_size]
                gene_weights = torch.as_tensor(gene_weights, dtype=torch.float32)

        self.trainer = Trainer(
            diffusion_model,
            self.ProcessData_path,
            self.concept_list,
            iteration=self.iteration,
            profile_size=training_profile_size,
            train_batch_size=training_batch_size,
            train_lr=training_lr,
            train_num_steps=epoch_num,
            results_folder=self.output_path,
            train_log=train_log,
            ema_decay=self.ema_decay,
            gradient_accumulate_every=self.gradient_accumulate_every,
            fp16=self.fp16,
            step_start_ema=self.step_start_ema,
            update_ema_every=self.update_ema_every,
            save_and_sample_every=self.save_and_sample_every,
            gene_weights=gene_weights,
        )
        self.model = diffusion_model

    def register_CPO_parameters(
        self,
        anchor_neighbors=15,
        max_neighbors=35,
        min_neighbors=10,
        resolution_min=0.8,
        resolution_max=1.5,
    ):
        self.CPO_parameters = {
            "anchor_neighbors": anchor_neighbors,
            "max_neighbors": max_neighbors,
            "min_neighbors": min_neighbors,
            "resolution_min": resolution_min,
            "resolution_max": resolution_max,
        }

    def register_species(self, species):
        normalized = {"human": "Human", "Human": "Human", "mouse": "Mouse", "Mouse": "Mouse"}
        if species not in normalized:
            raise ValueError("species must be Human or Mouse.")
        self.species = normalized[species]

    def register_iDREM_parameters(
        self,
        Normalize_data="Log_normalize_data",
        Minimum_Absolute_Log_Ratio_Expression=0.5,
        Convergence_Likelihood=0.001,
        Minimum_Standard_Deviation=0.5,
    ):
        allowed = {"Log_normalize_data", "Normalize_data", "No_normalize_data"}
        if Normalize_data not in allowed:
            raise ValueError(f"Normalize_data must be one of {sorted(allowed)}.")
        self.iDREM_parameters = {
            "Normalize_data": Normalize_data,
            "Minimum_Absolute_Log_Ratio_Expression": Minimum_Absolute_Log_Ratio_Expression,
            "Convergence_Likelihood": Convergence_Likelihood,
            "Minimum_Standard_Deviation": Minimum_Standard_Deviation,
        }

    def run_caunagi(self, idrem_dir, CPO=True):
        """Run one complete CauNagi iteration."""
        if self.trainer is None or self.iteration is None:
            raise RuntimeError("Call process_data and setup_train before run_caunagi.")
        iteration_dir = self.temp_path / str(self.iteration)
        if iteration_dir.exists():
            raise FileExistsError(
                f"The iteration directory already exists: {iteration_dir}"
            )
        (iteration_dir / "stagedata").mkdir(parents=True)

        runner = Caunagi_runner(
            self.data_path,
            self.temp_path,
            self.total_stage_num,
            self.iteration,
            self.trainer,
            idrem_dir,
            self.concept_list,
            self.concept_cdag,
            self.concept_counts,
            self.training_profile_size,
            self.CellType_dicts,
        )
        runner.set_up_species(self.species)
        if self.CPO_parameters is not None:
            runner.set_up_CPO(**self.CPO_parameters)
        if self.iDREM_parameters is not None:
            runner.set_up_IDREM(
                Minimum_Absolute_Log_Ratio_Expression=self.iDREM_parameters[
                    "Minimum_Absolute_Log_Ratio_Expression"
                ],
                Convergence_Likelihood=self.iDREM_parameters["Convergence_Likelihood"],
                Minimum_Standard_Deviation=self.iDREM_parameters[
                    "Minimum_Standard_Deviation"
                ],
            )
        runner.run(CPO)
        self.iteration += 1
        print("CauNagi iteration completed successfully.")

    def analyse_UNAGI(
        self,
        data_path,
        iteration,
        progressionmarker_background_sampling_times=1000,
        *,
        target_dir=None,
        save_dir=None,
        ignore_hcmarkers=False,
        ignore_dynamic_markers=False,
    ):
        """Run hierarchical and progression marker analyses.

        ``data_path`` may be the final ``stagedata`` directory or its
        ``dataset.h5ad`` file. Results are stored in ``save_dir``/``target_dir``
        or alongside the staged dataset by default.
        """
        if save_dir is not None and target_dir is not None:
            raise ValueError("Specify only one of save_dir and target_dir.")
        output_dir = save_dir or target_dir
        analyst = Analyst(data_path, iteration, target_dir=output_dir)
        result = analyst.start_analyse(
            progressionmarker_background_sampling_times,
            ignore_hcmarkers=ignore_hcmarkers,
            ignore_dynamic_markers=ignore_dynamic_markers,
        )
        print("CauNagi marker analyses completed successfully.")
        return result
