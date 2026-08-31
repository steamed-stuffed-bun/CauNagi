"""Training and latent-space utilities for the CauNagi diffusion model."""

import copy
import os
from functools import partial
from pathlib import Path

import numpy as np
import scanpy as sc
import torch
from scipy.spatial.distance import pdist
from sklearn.neighbors import NearestNeighbors
from torch.optim import Adam
from torch.utils import data

from .Module import Denoise_net, EMA, GaussianDiffusion, cycle, get_logger, loss_backwards
from .utils import Dataset

try:
    from apex import amp
except ImportError:
    amp = None


class Trainer:
    """Train CauNagi and expose its concept representations."""

    def __init__(
        self,
        diffusion_model,
        folder,
        factor_list,
        *,
        ema_decay=0.995,
        iteration=1,
        profile_size=200,
        train_batch_size=32,
        train_lr=2e-5,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        fp16=False,
        step_start_ema=2000,
        update_ema_every=1000,
        save_and_sample_every=100,
        results_folder="./results",
        train_log=True,
        gene_weights=None,
    ):
        self.factor_list = list(factor_list)
        self.model = diffusion_model
        self.device = next(diffusion_model.parameters()).device
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.iteration = iteration
        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every
        self.batch_size = train_batch_size
        self.train_profile_size = profile_size
        self.Dataset_profile_size = diffusion_model.profile_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        adata = sc.read_h5ad(folder)
        self.ds = Dataset(adata, self.Dataset_profile_size, self.factor_list)
        self.dl = cycle(
            data.DataLoader(
                self.ds,
                batch_size=train_batch_size,
                shuffle=True,
                pin_memory=self.device.type == "cuda",
            )
        )
        self.opt = Adam(self.model.parameters(), lr=train_lr)
        self.step = 0

        self.gene_weights = None
        if gene_weights is not None:
            self.gene_weights = torch.as_tensor(gene_weights, dtype=torch.float32).cpu()
            if self.gene_weights.ndim != 2 or self.gene_weights.shape[1] != self.Dataset_profile_size:
                raise ValueError(
                    "gene_weights must have shape (number_of_cells, profile_size)."
                )

        if fp16 and amp is None:
            raise ImportError("Install NVIDIA Apex to enable fp16 training.")
        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize(
                [self.model, self.ema_model], self.opt, opt_level="O1"
            )

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)
        self.train_log = train_log
        if train_log:
            self.logger = get_logger(str(self.results_folder / "training.log"))
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
        else:
            self.ema.update_model_average(self.ema_model, self.model)

    def _checkpoint_path(self, iteration, milestone):
        return self.results_folder / f"{iteration}-model-{milestone}.pt"

    def save(self, milestone):
        checkpoint = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
        }
        torch.save(checkpoint, self._checkpoint_path(self.iteration, milestone))
        print(f"Saved checkpoint to {self.results_folder}.")

    def load(self, milestone, iteration=None, resume_step=True):
        """Load a checkpoint and return the stored training step."""
        checkpoint_iteration = self.iteration if iteration is None else iteration
        checkpoint_path = self._checkpoint_path(checkpoint_iteration, milestone)
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.ema_model.load_state_dict(checkpoint.get("ema", checkpoint["model"]))
        if resume_step:
            self.step = int(checkpoint.get("step", milestone * self.save_and_sample_every))
        return self.step

    def _load_resume_checkpoint(self):
        final_milestone = self.train_num_steps // self.save_and_sample_every
        current = self._checkpoint_path(self.iteration, final_milestone)
        previous = self._checkpoint_path(self.iteration - 1, final_milestone)
        if current.is_file():
            self.load(final_milestone, iteration=self.iteration, resume_step=True)
            return
        if self.iteration > 0 and previous.is_file():
            self.load(final_milestone, iteration=self.iteration - 1, resume_step=False)
            self.step = 0

    def train(self):
        """Train until ``train_num_steps`` optimizer steps are complete."""
        self._load_resume_checkpoint()
        backwards = partial(loss_backwards, self.fp16)

        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                expressions, factors, weights, batch_indices = next(self.dl)
                expressions = torch.as_tensor(expressions, dtype=torch.float32, device=self.device)
                factors = torch.as_tensor(factors, dtype=torch.long, device=self.device)
                weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)

                batch_gene_weights = None
                if self.iteration > 0 and self.gene_weights is not None:
                    indices = torch.as_tensor(batch_indices, dtype=torch.long)
                    if torch.any(indices >= len(self.gene_weights)):
                        raise IndexError("A training cell index exceeds gene_weights.")
                    batch_gene_weights = self.gene_weights[indices].to(self.device)

                losses = self.model(
                    expressions,
                    factors,
                    weights=weights,
                    gene_weights=batch_gene_weights,
                )
                loss_recon, mask_recon_loss, loss_pred_o, loss_discriminator, prior_kl = losses
                loss = loss_recon + mask_recon_loss + loss_pred_o + loss_discriminator + prior_kl

                if self.train_log:
                    self.logger.info(
                        "%s:%s loss_recon:%s mask_recon:%s pred_o:%s "
                        "discriminator:%s prior_kl:%s",
                        self.step,
                        _,
                        loss_recon.item(),
                        mask_recon_loss.item(),
                        loss_pred_o.item(),
                        loss_discriminator.item(),
                        prior_kl.item(),
                    )
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad(set_to_none=True)
            if self.step % self.update_ema_every == 0:
                self.step_ema()
            if self.step and self.step % self.save_and_sample_every == 0:
                self.save(self.step // self.save_and_sample_every)
            self.step += 1
        print("Training completed.")

    def load_trained(self, concept_list, concept_counts, concept_cdag, timesteps=1000):
        """Register concept metadata and return the already-trained model.

        The previous implementation reconstructed a second model and assumed
        a fixed early checkpoint. The trainer already owns the trained model,
        so reusing it avoids device and checkpoint mismatches.
        """
        self.concept_list = list(concept_list)
        self.concept_counts = list(concept_counts)
        self.concept_cdag = concept_cdag
        self.model.eval()
        return self.model

    def sampling_concepts(self, adata, concept_list, concept_counts, concept_cdag, profile_size=1000):
        """Sample disentangled concept embeddings for every cell."""
        dataset = Dataset(adata, profile_size, concept_list)
        loader = data.DataLoader(dataset, batch_size=1280, shuffle=False)
        batches = []
        encoder = self.model.denosie_fn.DisentanglementEncoder
        was_training = encoder.training
        encoder.eval()
        with torch.no_grad():
            for expressions, labels, _, _ in loader:
                embeddings = encoder(
                    expressions.to(self.device, dtype=torch.float32),
                    labels.to(self.device, dtype=torch.long),
                )[0]
                batches.append(embeddings.cpu())
        encoder.train(was_training)
        if not batches:
            return np.empty((len(concept_list) + 1, 0, 32), dtype=np.float32)
        embeddings = torch.cat(batches, dim=0).numpy()
        return np.transpose(embeddings, (1, 0, 2))

    def disentanglement(self, adata, sampling_counts=10):
        if sampling_counts < 1:
            raise ValueError("sampling_counts must be positive.")
        samples = [
            self.sampling_concepts(
                adata,
                self.concept_list,
                self.concept_counts,
                self.concept_cdag,
                profile_size=self.train_profile_size,
            )
            for _ in range(sampling_counts)
        ]
        return np.mean(samples, axis=0)

    def generate_concept_spaces(self, adata):
        """Estimate Gaussian concept spaces from repeated encoder samples."""
        concept_embeddings = self.disentanglement(adata)
        n_concepts, n_cells, n_features = concept_embeddings.shape
        concept_spaces = {}
        for concept in range(n_concepts):
            embedding = concept_embeddings[concept]
            if n_cells < 3:
                scales = np.full((n_cells, n_features), 1e-3, dtype=np.float32)
            else:
                n_neighbors = min(50, n_cells - 1)
                indices = NearestNeighbors(n_neighbors=n_neighbors).fit(embedding).kneighbors(
                    embedding, return_distance=False
                )
                variances = np.zeros(n_cells, dtype=np.float32)
                for index, neighbors in enumerate(indices):
                    distances = pdist(embedding[neighbors], metric="cosine")
                    variances[index] = float(np.nanmean(distances)) if len(distances) else 0.0
                scales = np.maximum(variances[:, None], 1e-6)
                scales = np.repeat(scales, n_features, axis=1).astype(np.float32)
            scales = np.sqrt(scales)
            concept_spaces[concept] = (embedding, scales, embedding + np.log(scales))
        return concept_spaces

    def get_latent_representation(self, adata, concept_key="CellType"):
        if concept_key not in self.concept_list:
            raise KeyError(f"Unknown concept: {concept_key}")
        concept_space = self.generate_concept_spaces(adata)[self.concept_list.index(concept_key)]
        return concept_space
