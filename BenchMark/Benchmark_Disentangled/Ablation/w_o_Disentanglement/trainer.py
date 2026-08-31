import copy
from .Module import EMA,cycle,get_logger,loss_backwards
from .utils import Dataset,transfer_to_ranking_score
from torch.utils import data
from torch.optim import Adam
from pathlib import Path
import torch
from functools import partial
from copy import deepcopy
from .Module import Denoise_net, GaussianDiffusion
import numpy as np
import scipy.sparse as sp
import scanpy as sc
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")

try:
    from apex import amp
    APEX_AVAILABLE = True
except:
    APEX_AVAILABLE = False

from sklearn.neighbors import NearestNeighbors
from scipy.spatial.distance import pdist, squareform

class Trainer(object):
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
            results_folder='./results',
            train_log=True,
            gene_weights = None
    ):
        super().__init__()
        self.factor_list = factor_list
        self.model = diffusion_model
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
        # print(adata.X.shape)
        self.ds = Dataset(adata, self.Dataset_profile_size, factor_list)
        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0
        
        ##1
        self.gene_weights = gene_weights
        
        assert not fp16 or fp16 and APEX_AVAILABLE, 'Apex must be installed in order for mixed precision training to be turned on'

        self.fp16 = fp16
        if fp16:
            (self.model, self.ema_model), self.opt = amp.initialize([self.model, self.ema_model], self.opt,
                                                                    opt_level='O1')

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        if train_log:
            self.logger = get_logger(results_folder + '/training.log')
        self.train_log = train_log
        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            'step': self.step,
            'model': self.model.state_dict(),
            'ema': self.ema_model.state_dict()
        }
        torch.save(data, str(self.results_folder / f'{self.iteration}-model-{milestone}.pt'))
        print("The model is saved as %s" % self.results_folder)

    def load(self, milestone):
        torch.save(data, str(self.results_folder / f'{self.iteration}-model-{milestone}.pt'))

        self.step = data['step']
        self.model.load_state_dict(data['model'])
        self.ema_model.load_state_dict(data['ema'])

    def train(self):
        milestone = self.train_num_steps // self.save_and_sample_every
        if os.path.exists(str(Path(self.results_folder)) +'/'+str(self.iteration-1)+'-model-' + str(milestone) + '.pt'):
            if os.path.exists(str(Path(self.results_folder)) +'/'+str(self.iteration)+'-model-' + str(milestone) + '.pt'):
                print('load current iteration model....')
                data = torch.load(str(Path(self.results_folder)) + '/' + str(self.iteration) + '-model-' + str(milestone) + '.pt')
                self.model.load_state_dict(data['model'])
            else:
                print('load last iteration model.....')
                data = torch.load(str(Path(self.results_folder)) + '/' + str(self.iteration-1) + '-model-' + str(milestone) + '.pt')
                self.model.load_state_dict(data['model'])
                
                
        
        backwards = partial(loss_backwards, self.fp16)

        while self.step <= self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                expressions, factors, weights, batch_indices = next(self.dl)
                
                print(expressions.shape)
                expressions = torch.tensor(expressions, dtype=torch.float32)
                factors = torch.tensor(factors, dtype=torch.long)  # 根据实际情况调整类型
                weights = torch.tensor(weights, dtype=torch.float32)
                
                
                expressions = expressions.cuda()
                factors = factors.cuda()
                weights = weights.cuda()
                # inputs = data[0].cuda().float()    # 转换为 float32
                # labels = data[1].cuda()
                # weights = data[2].cuda().float()    # 转换为 float32
                # print(inputs.dtype,labels.dtype,weights.dtype)
                # loss_recon, mask_recon_loss, loss_pred_o, loss_discriminator, prior_kl = self.model(inputs, labels, weights)
                
                is_iterative = (self.iteration != 0)
                # is_iterative = True
                batch_gene_weights = None
                if is_iterative and self.gene_weights is not None:
                    # 确保索引有效
                    batch_indices = [idx for idx in batch_indices if idx < len(self.gene_weights)]
                    if batch_indices:
                        if isinstance(batch_indices, list):
                            # 把 [tensor(1166), tensor(6076), ...] 变成 tensor([1166, 6076, ...])
                            batch_indices = torch.tensor([idx.item() for idx in batch_indices], 
                                                         dtype=torch.long, 
                                                         device=self.gene_weights.device)
                        else:
                            batch_indices = batch_indices.long().to(self.gene_weights.device)
                            
                        batch_gene_weights = self.gene_weights[batch_indices]
                        batch_gene_weights = batch_gene_weights.to(expressions.device)
                
                loss_recon, mask_recon_loss, loss_pred_o, loss_discriminator, prior_kl = self.model(
                    expressions, 
                    factors, 
                    weights=weights,  # 原版权重
                    gene_weights=batch_gene_weights  # UNAGI基因权重
                )
                
                
                #loss = loss_recon + loss_pred_o + loss_discriminator + prior_kl
                loss = loss_recon + prior_kl
                # loss = loss_recon
                # loss = loss_pred_o
                if self.train_log:
                    self.logger.info(
                        f'{self.step}:{i}\tloss_recon:{loss_recon.item()}\tloss_pred_o:{loss_pred_o.item()}\tloss_discriminator:{loss_discriminator.item()}\tprior_kl:{prior_kl.item()}')
                # self.logger.info(f'{self.step}:{i}\tloss_recon:{loss_recon.item()}\tloss_pred_o:{loss_pred_o.item()}\tprior_kl:{prior_kl.item()}')
                # print(f'{self.step}:{loss_recon.item()}')
                # print(f'{self.step}:{loss_pred_o.item()}')
                # print(f'{self.step}:{loss_discriminator.item()}')
                # print(f'{self.step}:{prior_kl.item()}')
                # print(f'{self.step}:{loss.item()}')
                backwards(loss / self.gradient_accumulate_every, self.opt)

            self.opt.step()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()
            # print("save_and_sample_every is", self.save_and_sample_every)
            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
                print("The model is saved")
            # else:
            #     print("step is %s"%self.step)
            #     print("step this variable is not equal to 0?\n",self.step != 0)
            #     print("step % save_and_sample_every is equal to 0?\n",self.step % self.save_and_sample_every == 0)
            self.step += 1
        print('training completed')



    def load_trained(self,
                    concept_list,
                    concept_counts,
                    concept_cdag,
                    timesteps=1000):

        # Initialization parameters
        # self.results_folder = save_path
        milestone =  self.train_num_steps // self.save_and_sample_every
        # concept list
        self.concept_list = concept_list
        self.concept_counts = concept_counts
        self.concept_cdag = concept_cdag
        print(type(self.concept_counts),self.concept_counts)
        factor_order = concept_list
        profile_size = self.train_profile_size
        model_denoise = Denoise_net(profile_size, profile_size, len(concept_counts) + 1,
                                    torch.tensor(concept_cdag).float(), concept_counts).cuda()
        GaussianDiffusion_model = GaussianDiffusion(model_denoise, profile_size=profile_size,
                                                    timesteps=timesteps).cuda()
        data = torch.load(str(Path(self.results_folder)) +'/'+str(self.iteration)+'-model-' + str(milestone) + '.pt')
        GaussianDiffusion_model.load_state_dict(data['model'])
        self.model = deepcopy(GaussianDiffusion_model)


    def sampling_concepts(self,
                          adata,
                          concept_list,
                          concept_counts,
                          concept_cdag,
                          profile_size=1000):
        with torch.no_grad():
            # adata.X = adata.X.astype(np.float32)
            dataset = Dataset(adata, profile_size, concept_list)
            dataloader = data.DataLoader(dataset, batch_size=1280, shuffle=False, pin_memory=True)
            disentanglement_embs = []
            for idx, data_ in enumerate(dataloader):
                batch_concept_embs = \
                self.model.denosie_fn.DisentanglementEncoder.eval()(data_[0].cuda(), data_[1].cuda())[0].cpu()
                disentanglement_embs.append(batch_concept_embs)

                # batch_concept_embs = self.model.denosie_fn.DisentanglementEncoder.eval()(
                #     data_[0].cuda().float(),
                #     data_[1].cuda().float()
                # )[0].cpu()
                # input_data = data_[0].cuda().to(self.model.dtype)  # 确保数据类型一致
                # other_data = data_[1].cuda().to(self.model.dtype)  # 确保数据类型一致
                # batch_concept_embs = self.model.denosie_fn.DisentanglementEncoder.eval()(input_data, other_data)[0].cpu()
                
            concept_embs = [[] for _ in range(len(concept_list) + 1)]
            for factor in range(len(concept_list) + 1):
                for idx, i in enumerate(disentanglement_embs):
                    for b in i:
                        concept_embs[factor].append(b[factor])

            concept_embs_stacked = []
            for i in concept_embs:
                concept_embs_stacked.append(np.array(torch.stack(i)))
        return np.array(concept_embs_stacked)


    def disentanglement(self,
                        adata,
                        sampling_counts=10):
        # concept list
        concept_list = self.concept_list
        concept_counts = self.concept_counts
        concept_cdag = self.concept_cdag

        # saved_path = self.output_path

        profile_size = self.train_profile_size

        concept_embs = self.sampling_concepts(adata, concept_list, concept_counts, concept_cdag,
                                              profile_size=profile_size)
        for i in range(sampling_counts - 1):
            concept_embs += self.sampling_concepts(adata, concept_list, concept_counts, concept_cdag,
                                                   profile_size=profile_size)
        concept_embs /= sampling_counts
        # if not os.path.exists(saved_path):
        #     os.mkdir(saved_path)
      # joblib.dump(concept_embs, saved_path + f"/factors_embs.pkl")
        return concept_embs

    def generate_concept_spaces(self, adata):
        """
        为Causcell的每个概念生成UNAGI兼容的潜在空间表示
    
        参数:
            concept_embs: (n_concepts, n_cells, n_features) 三维张量
    
        返回:
            concept_spaces: 字典，key为概念索引，值为 (z_locs, z_scales, TZ)
        """
        concept_embs = self.disentanglement(adata)
        n_concepts, n_cells, n_features = concept_embs.shape
        concept_spaces = {}
    
        for c in range(n_concepts):
            # 提取当前概念的细胞嵌入
            emb = concept_embs[c]  # (n_cells, n_features)
    
            # 1. z_locs: 当前概念的细胞嵌入
            z_locs = emb
    
            # 2. z_scales: 基于空间一致性的不确定性度量
            # 计算每个细胞的空间邻域内嵌入的变异度
            n_neighbors = min(50, n_cells // 10)  # 自适应邻居数
            knn = NearestNeighbors(n_neighbors=n_neighbors).fit(emb)
            _, indices = knn.kneighbors(emb)
    
            neighbor_variance = np.zeros(n_cells)
            for i in range(n_cells):
                # 当前细胞的邻居嵌入
                neighbor_embs = emb[indices[i]]
                # 计算邻居间的平均余弦距离（差异越大→不确定性越高）
                pairwise_dists = pdist(neighbor_embs, 'cosine')
                neighbor_variance[i] = np.mean(pairwise_dists)
    
            # 转换为对数标准差（遵循UNAGI分布特性）
            z_scales = neighbor_variance.reshape(-1, 1)  # (n_cells, 1)
            z_scales = np.log(z_scales + 1e-6)  # 取对数
            z_scales = np.tile(z_scales, (1, n_features))  # 扩展到特征维度
    
            # 3. TZ: 生成与UNAGI完全兼容的合成编码
            TZ = z_locs + z_scales  # 使用相同形式 z_locs + log(sigma)
    
            z_locs = np.array(z_locs)
            z_scales = np.array(z_scales)
            z_scales = np.exp(0.5 * z_scales)
            TZ = np.array(TZ)
    
            concept_spaces[c] = (z_locs, z_scales, TZ)
    
        return concept_spaces

    def get_latent_representation(self,adata,concept_key = "CellType"):
        concept_list = self.concept_list
        concept_spaces = self.generate_concept_spaces(adata)
        concept_key_idx = concept_list.index(concept_key)
        concept_key_space = concept_spaces[concept_key_idx]
        return concept_key_space[0],concept_key_space[1],concept_key_space[2]

