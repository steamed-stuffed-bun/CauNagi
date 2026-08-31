import math
import os
from inspect import getfullargspec, isfunction
from typing import Any, List, Optional

import anndata
import numpy as np
import scanpy as sc
import torch
import torch.nn as nn
from scipy.stats import rankdata
from torch.utils import data
def transfer_to_ranking_score(gw):
    '''
    transfer the gene weight to ranking
    '''
    # gw = adata.layers['geneWeight'].toarray()
    od = gw.shape[1]-rankdata(gw,axis=1)+1
    score = 1+1/np.power(od,0.5)
    
    return score


def merge_data(path,total_stage):
    path = os.fspath(path)
    stage_datas = []
    for i in range(total_stage):
        data_path = os.path.join(path,'%d.h5ad'%i)
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"Stage data not found: {data_path}")
        temp_data = sc.read_h5ad(data_path)
        if i and not temp_data.var_names.equals(stage_datas[0].var_names):
            raise ValueError("All stages must use the same ordered gene set.")
        stage_datas.append(temp_data)
    adata_combined = anndata.concat(stage_datas, axis=0, join='inner', merge='same')
    adata_combined.write_h5ad(os.path.join(path, 'total_data.h5ad'), compression='gzip', compression_opts=9)
    return adata_combined

def make_beta_schedule(schedule, n_timestep, linear_start=1e-4, linear_end=2e-2, cosine_s=8e-3):
    if schedule == "linear":
        betas = (
                torch.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep, dtype=torch.float64) ** 2
        )

    elif schedule == "cosine":
        timesteps = (
                torch.arange(n_timestep + 1, dtype=torch.float64) / n_timestep + cosine_s
        )
        alphas = timesteps / (1 + cosine_s) * np.pi / 2
        alphas = torch.cos(alphas).pow(2)
        alphas = alphas / alphas[0]
        betas = 1 - alphas[1:] / alphas[:-1]
        betas = np.clip(betas, a_min=0, a_max=0.999)

    elif schedule == "sqrt_linear":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64)
    elif schedule == "sqrt":
        betas = torch.linspace(linear_start, linear_end, n_timestep, dtype=torch.float64) ** 0.5
    else:
        raise ValueError(f"schedule '{schedule}' unknown.")
    return betas.numpy()


def split_dataset_into_stage(data_path, folder, key):
    if not os.path.exists(data_path):
        raise FileNotFoundError('The specified path does not exist, Please check the path again.')
    if not os.path.exists(folder):
        raise FileNotFoundError('The specified path does not exist in this folder.')
    adata = sc.read_h5ad(data_path)
    adata_key = adata.obs.columns
    if key not in adata_key:
        raise ValueError('This keyword does not exist in the data set.')
    for each in list(adata.obs[key].unique()):
        adata_temp = adata[adata.obs[key] == each]
        # if 'X_pca' not in adata_temp.obsm.keys():
        #     sc.tl.pca(adata_temp)
        adata_temp.write_h5ad(os.path.join(folder, 'stage_' + '%s.h5ad' % each), compression='gzip', compression_opts=9)
    return adata_temp.shape[1]


def create_activation(name):
    if name is None:
        return nn.Identity()
    elif name == "relu":
        return nn.ReLU()
    elif name == "gelu":
        return nn.GELU()
    elif name == "glu":
        return nn.GLU()
    elif name == "sigmoid":
        return nn.Sigmoid()
    elif name == "prelu":
        return nn.PReLU()
    elif name == "elu":
        return nn.ELU()
    else:
        raise NotImplementedError(f"{name} is not implemented.")


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class BatchedOperation:
    """Wrapper to expand batched dimension for input tensors.

    Args:
        batch_dim: Which dimension the batch goes.
        plain_num_dim: Number of dimensions for plain (i.e., no batch) inputs,
            which is used to determine whether the input the batched or not.
        ignored_args: Which arguments to ignored for automatic batch dimension
            expansion.
        squeeze_output_batch: If set to True, then try to squeeze out the batch
            dimension of the output tensor.

    """

    def __init__(
            self,
            batch_dim: int = 0,
            plain_num_dim: int = 2,
            ignored_args: Optional[List[str]] = None,
            squeeze_output_batch: bool = True,
    ):
        self.batch_dim = batch_dim
        self.plain_num_dim = plain_num_dim
        self.ignored_args = set(ignored_args or [])
        self.squeeze_output_batch = squeeze_output_batch
        self._is_batched = None

    def __call__(self, func):
        arg_names = getfullargspec(func).args

        def bounded_func(*args, **kwargs):
            new_args = []
            for arg_name, arg in zip(arg_names, args):
                if self.unsqueeze_batch_dim(arg_name, arg):
                    arg = arg.unsqueeze(self.batch_dim)
                new_args.append(arg)

            for arg_name, arg in kwargs.items():
                if self.unsqueeze_batch_dim(arg_name, arg):
                    kwargs[arg_name] = arg.unsqueeze(self.batch_dim)

            out = func(*new_args, **kwargs)

            if self.squeeze_output_batch:
                out = out.squeeze(self.batch_dim)

            return out

        return bounded_func

    def unsqueeze_batch_dim(self, arg_name: str, arg_val: Any) -> bool:
        return (
                isinstance(arg_val, torch.Tensor)
                and (arg_name not in self.ignored_args)
                and (not self.is_batched(arg_val))
        )

    def is_batched(self, val: torch.Tensor) -> bool:
        num_dim = len(val.shape)
        if num_dim == self.plain_num_dim:
            return False
        elif num_dim == self.plain_num_dim + 1:
            return True
        else:
            raise ValueError(
                f"Tensor should have either {self.plain_num_dim} or "
                f"{self.plain_num_dim + 1} number of dimension, got {num_dim}",
            )


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def exists(x):
    return x is not None


def create_norm(name, n, h=16):
    if name is None:
        return nn.Identity()
    elif name == "layernorm":
        return nn.LayerNorm(n)
    elif name == "batchnorm":
        return nn.BatchNorm1d(n)
    elif name == "groupnorm":
        return nn.GroupNorm(h, n)
    elif name.startswith("groupnorm"):
        inferred_num_groups = int(name.replace("groupnorm", ""))
        return nn.GroupNorm(inferred_num_groups, n)
    else:
        raise NotImplementedError(f"{name} is not implemented.")


def extract_into_tensor(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))


class Dataset(data.Dataset):
    def __init__(self,adata, profile_size, factor_list):
        super().__init__()
        self.profile_size = profile_size
        if profile_size > adata.n_vars:
            raise ValueError("profile_size cannot exceed the number of genes.")
        if not set(factor_list).issubset(adata.obs.columns):
            missing = set(factor_list).difference(adata.obs.columns)
            raise KeyError("Missing factor columns: " + ", ".join(sorted(missing)))

        expression = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
        self.profile_data = np.asarray(expression[:, :profile_size], dtype=np.float32)
        self.labels = adata.obs.loc[:, list(factor_list)].reset_index(drop=True)

        tuple_list = [tuple(row) for row in self.labels.to_numpy()]
        unique_tuplpe_list = list(dict.fromkeys(tuple_list))
        tmp_dict = dict(zip(unique_tuplpe_list, range(len(unique_tuplpe_list))))
        merged_class = [tmp_dict[i] for i in tuple_list]

        freq = {}
        for item in merged_class:
            freq[item] = freq.get(item, 0) + 1

        self.weights = np.array([1 - (freq[i] / len(self.labels)) for i in merged_class], dtype=np.float32)
        
        self.cell_indices = np.arange(len(data)) 
    def __len__(self):
        return len(self.profile_data)

    def __getitem__(self, index):
        cell_exp = self.profile_data[index]
        labels = self.labels.iloc[index].to_numpy(dtype=np.int64)
        weight = self.weights[index]
        
        cell_index = self.cell_indices[index]
        return cell_exp, labels, weight, cell_index 

def noise_like(shape, device, repeat=False):
    if repeat:
        noise = torch.randn((1, *shape[1:]), device=device)
        repeat_noise = noise.repeat(shape[0], *((1,) * (len(shape) - 1)))
        return repeat_noise
    else:
        return torch.randn(shape, device=device)

def convert_uns_keys_to_str(uns_dict):
    new_dict = {}
    for k, v in uns_dict.items():
        key = str(k) if isinstance(k, (int, np.integer)) else k
        new_dict[key] = convert_uns_keys_to_str(v) if isinstance(v, dict) else v
    return new_dict
