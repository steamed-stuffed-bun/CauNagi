import torch
import numpy as np
import pandas as pd
import anndata as ad
import random
from .Module import Denoise_net, GaussianDiffusion
from .trainer import Trainer
import scanpy as sc
import joblib
from copy import deepcopy
from scipy.sparse import csr_matrix, issparse
import os
import subprocess
# os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from .utils import split_dataset_into_stage, Dataset ,merge_data
from .runner import Caunagi_runner
import scipy.sparse as sp
from .get_driver import analyst
# from scipy.sparse import csr_matrix
# import scipy.sparse as sp
# from sklearn.neighbors import kneighbors_graph
# import time
# from torch.distributions.normal import Normal
# import threading
# from pathlib import Path
# from torch.utils import data

class Caunagi:
    """
    Caunagi base model class
    """

    def __init__(self,
                 concept_list,
                 concept_cdag,
                 total_stage_num,
                 device='cuda',
                 ema_decay=0.995,
                 gradient_accumulate_every=2,
                 fp16=False,
                 step_start_ema=2000,
                 update_ema_every=1000,
                 save_and_sample_every=100
                 ):

        self.device = device
        self.ema_decay = ema_decay
        self.gradient_accumulate_every = gradient_accumulate_every
        self.fp16 = fp16
        self.step_start_ema = step_start_ema
        self.update_ema_every = update_ema_every
        self.save_and_sample_every = save_and_sample_every
        self.model = None
        self.CPO_parameters = None
        self.iDREM_parameters = None
        self.species = 'Human'
        self.concept_list = concept_list
        self.concept_cdag = concept_cdag
        self.total_stage_num = total_stage_num

    def process_data(self,
                     data_path,
                     temp_path,
                     stage_name,
                     iteration,
                     celltype_concept_name,
                     disease_idx,
                     log_norm=False):

        self.data_path = data_path
        self.temp_path = temp_path
        self.stage_name = stage_name
        self.iteration = iteration

        concept_list = self.concept_list
        concept_cdag = self.concept_cdag

        # if os.path.isdir(data_path):
        #     raise ValueError('The path you provided points to a folder rather than a file.')
        # if os.path.exists(temp_path):
        #     raise ValueError('The temp folder already exists. Please delete the folder before running this script.')

        # input_data = sc.read_h5ad(data_path)
        if self.iteration == 0:
            if not os.path.exists(temp_path):
                os.makedirs(temp_path)
            else:
                raise ValueError('The temp folder already exists. Please delete the folder before running this script.')
            input_data = merge_data(data_path,self.total_stage_num)
        else:
            input_data = sc.read_h5ad(os.path.join(data_path,"%d/stagedata"%(iteration-1))+'/dataset.h5ad')
        
        
        if issparse(input_data.X):
            input_data.X = input_data.X.toarray()
        # print("Is sparse_matrix sparse?", issparse(input_data.X)) 


        if log_norm:
            normed_data = input_data.X / input_data.X.sum(axis=1)[:, None] * 10000
            exp_data = np.log(normed_data + 1)
            exp_data = exp_data.astype(np.float32)
        else:
            exp_data = input_data.X
            exp_data = exp_data.astype(np.float32)
        add_obs = []
        label_categories = []
        for idx, factor_name in enumerate(concept_list):
            if factor_name != stage_name:
                factor_vals = list(input_data.obs[factor_name].unique())
                val2idx = dict(zip(factor_vals, range(len(factor_vals))))
                add_obs.append([val2idx[i] for i in input_data.obs[factor_name]])
                label_categories.append(len(factor_vals))
                joblib.dump(val2idx, f"{data_path}/{factor_name}_dict.pkl")
                if factor_name == celltype_concept_name:
                     self.CellType_dicts = {key: value for key, value in val2idx.items()}

            else:
                disease_idx = disease_idx
                add_obs.append([disease_idx[i] for i in input_data.obs[factor_name]])
                label_categories.append(len(list(disease_idx.keys())))
                joblib.dump(disease_idx, f"{data_path}/{factor_name}_dict.pkl")

        concep_df = pd.DataFrame(list(zip(*add_obs)), columns=concept_list)
        new_data = ad.AnnData(exp_data)
        new_data.obs = concep_df
        # data_name = data_path.split("/")[-1]
        self.ProcessData_path = os.path.join(data_path,'process_data.h5ad')
        new_data.write(self.ProcessData_path)
        self.input_data = input_data
        # self.concept_list = concept_list
        self.concept_counts = label_categories
        # self.concept_cdag = concept_cdag
        if len(concept_list) + 1 != len(concept_cdag):
            raise ValueError(
                'The dimensions of your concept adjacency matrix need to take into account unexplained concepts, so it must be one dimension larger than the concept list.')
        print("The input dataset has been processed.")

    # def load_data(self):
    #
    #     total_stage_num = self.total_stage_num
    #
    #     temp_path = self.temp_path
    #     data_path = self.ProcessData_path
    #     stage_name = self.stage_name
    #     if total_stage_num < 2:
    #         raise ValueError('The total number of stages should be larger than 1')
    #     self.stage_key = stage_name
    #     self.input_dim = split_dataset_into_stage(data_path, temp_path, stage_name)
    #     print('The dataset was divided into %s stages.' % total_stage_num)
    #     self.data_path = data_path
    #     self.ns = total_stage_num
    #     self.data_folder = temp_path

    def setup_train(self,
                    model_save_path,
                    *,
                    loss_type="l2",
                    epoch_num=100000,
                    training_batch_size=64,
                    training_lr=2e-5,
                    max_profile_size=2000,
                    timesteps=1000,
                    seed=888,
                    train_log=True):
        # random seed setting
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.cuda.manual_seed(seed)

        # concept list
        concept_list = self.concept_list
        concept_counts = self.concept_counts
        concept_cdag = self.concept_cdag

        # read training dataset
        train_data_path = self.ProcessData_path
        train_data = sc.read_h5ad(train_data_path)
        self.output_path = model_save_path
        if train_data.X.shape[1] > max_profile_size:
            training_profile_size = max_profile_size
        else:
            training_profile_size = train_data.X.shape[1]
        self.training_profile_size = training_profile_size
        self.epoch_num = epoch_num
        model = Denoise_net(training_profile_size, training_profile_size, len(concept_counts) + 1,
                            torch.tensor(concept_cdag).float(), concept_counts).to(self.device)
        diffusion_model = GaussianDiffusion(model, profile_size=training_profile_size, timesteps=timesteps,
                                            loss_type=loss_type).to(self.device)
                                            
        
        gene_weights = None
        # is_iterative = (self.iteration != 0)
        is_iterative = False
        if is_iterative and self.input_data is not None:
            """从Caunagi的AnnData对象加载基因权重"""
            if 'geneWeight' in self.input_data.layers:
                gene_weights = self.input_data.layers['geneWeight']
                
                # 处理稀疏矩阵
                if sp.issparse(gene_weights):
                    gene_weights = gene_weights.toarray()
                
                # 转换为PyTorch张量
                gene_weights = torch.tensor(gene_weights, dtype=torch.float32)
                print(f"成功加载Caunagi基因权重，形状: {gene_weights.shape}")
            else:
                print("警告: AnnData对象缺少geneWeight层")
                gene_weights = None
        
        
        self.trainer = Trainer(diffusion_model, train_data_path, concept_list,
                               iteration = self.iteration,
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
                               gene_weights = gene_weights
                               )

        # self.trainer.train()
        self.model = deepcopy(diffusion_model)

    def register_CPO_parameters(self, anchor_neighbors=15, max_neighbors=35, min_neighbors=10, resolution_min=0.8,
                                resolution_max=1.5):
        self.CPO_parameters = {}
        self.CPO_parameters['anchor_neighbors'] = anchor_neighbors
        self.CPO_parameters['max_neighbors'] = max_neighbors
        self.CPO_parameters['min_neighbors'] = min_neighbors
        self.CPO_parameters['resolution_min'] = resolution_min
        self.CPO_parameters['resolution_max'] = resolution_max

    def register_species(self, species):
        if species not in ['human', 'mouse', 'Human', 'Mouse']:
            raise ValueError('species should be either human or mouse')
        if species == 'human':
            species = 'Human'
        if species == 'mouse':
            species = 'Mouse'
        self.species = species

    def register_iDREM_parameters(self, Normalize_data='Log_normalize_data', Minimum_Absolute_Log_Ratio_Expression=0.5,
                                  Convergence_Likelihood=0.001, Minimum_Standard_Deviation=0.5):
        self.iDREM_parameters = {}
        if Normalize_data not in ['Log_normalize_data', 'Normalize_data', 'No_normalize_data']:
            raise ValueError(
                'Normalize_data should be chosen from Log_normalize_data, Normalize_data and No_normalize_data')
        self.iDREM_parameters['Normalize_data'] = Normalize_data
        self.iDREM_parameters['Minimum_Absolute_Log_Ratio_Expression'] = Minimum_Absolute_Log_Ratio_Expression
        self.iDREM_parameters['Convergence_Likelihood'] = Convergence_Likelihood
        self.iDREM_parameters['Minimum_Standard_Deviation'] = Minimum_Standard_Deviation

    def run_caunagi(self, idrem_dir, CPO=True):
        start_iteration = 0
        import json
        # with open(os.path.join(self.data_folder, 'model_save') + '/training_parameters.json', 'w') as json_file:
        #     json.dump(self.training_parameters, json_file, indent=4)
        iteration = self.iteration

        dir1 = os.path.join(self.temp_path, str(iteration))
        dir2 = os.path.join(self.temp_path, str(iteration) + '/stagedata')
        # dir3 = os.path.join(self.temp_path, 'model_save')
        initalcommand = 'mkdir ' + dir1 + ' && mkdir ' + dir2
        p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)


        self.adversarial = True
        runner = Caunagi_runner(self.data_path,
                                    self.temp_path,
                                    self.total_stage_num,
                                    iteration,
                                    self.trainer,
                                    idrem_dir,
                                    self.concept_list,
                                    self.concept_cdag,
                                    self.concept_counts,
                                    self.training_profile_size,
                                    self.CellType_dicts,
                                    adversarial=self.adversarial,
                                    )
        runner.set_up_species(self.species)
        if self.CPO_parameters is not None:
            if type(self.CPO_parameters) != dict:
                raise ValueError('CPO_parameters should be a dictionary')
            else:
                runner.set_up_CPO(anchor_neighbors=self.CPO_parameters['anchor_neighbors'],
                                        max_neighbors=self.CPO_parameters['max_neighbors'],
                                        min_neighbors=self.CPO_parameters['min_neighbors'],
                                        resolution_min=self.CPO_parameters['resolution_min'],
                                        resolution_max=self.CPO_parameters['resolution_max'])
        if self.iDREM_parameters is not None:
            if type(self.iDREM_parameters) != dict:
                raise ValueError('iDREM_parameters should be a dictionary')
            else:
                runner.set_up_IDREM(Minimum_Absolute_Log_Ratio_Expression=self.iDREM_parameters[
                    'Minimum_Absolute_Log_Ratio_Expression'],
                                          Convergence_Likelihood=self.iDREM_parameters['Convergence_Likelihood'],
                                          Minimum_Standard_Deviation=self.iDREM_parameters[
                                              'Minimum_Standard_Deviation'])
        runner.run(CPO)
        print("The model is success complete!")

    def analyse_UNAGI(self, data_path, iteration, progressionmarker_background_sampling_times, run_pertubration,
                      customized_pathway=None, target_dir=None, customized_drug=None, cmap_dir=None,
                      defulat_perturb_change=0.5, overall_perturbation_analysis=True, perturbed_tracks='all',
                      ignore_pathway_perturabtion=False, ignore_drug_perturabtion=False, centroid=False,
                      ignore_hcmarkers=False, ignore_dynamic_markers=False ,save_dir=None, model_path = None):
        if not os.path.exists(save_dir):
            raise ValueError('The folder is not exists. Please make sure the folder is exists before running this script.')
        analysts = analyst(data_path, iteration, target_dir=target_dir, customized_drug=customized_drug,
                           cmap_dir=cmap_dir, model_path = model_path)
        analysts.start_analyse(progressionmarker_background_sampling_times, customized_pathway=customized_pathway,
                               run_pertubration=run_pertubration,
                               random_times=progressionmarker_background_sampling_times,
                               defulat_perturb_change=defulat_perturb_change,
                               overall_perturbation_analysis=overall_perturbation_analysis,
                               perturbed_tracks=perturbed_tracks,
                               ignore_pathway_perturabtion=ignore_pathway_perturabtion,
                               ignore_drug_perturabtion=ignore_drug_perturabtion, centroid=centroid,
                               ignore_hcmarkers=ignore_hcmarkers, ignore_dynamic_markers=ignore_dynamic_markers,save_dir=save_dir,data_path=self.data_path)
        print('The analysis has been done, please check the outputs!')
