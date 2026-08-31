import gc
import os
import json
import pickle
import scanpy as sc
import subprocess
import numpy as np
from .analysis_helper import find_overlap_and_assign_direction, calculateDataPathwayOverlapGene, \
    calculateTopPathwayGeneRanking, process_customized_drug_database
from .hierachical_static_markers import get_dataset_hcmarkers
# from .perturbations.perturbation import perturbation
# from .perturbations.perturbation_centroid import perturbation as perturbation_centroid
from .dynamic_markers_helper import get_progressionmarker_background
from .dynamic_markers import runGetProgressionMarker_one_dist
from .utils import convert_uns_keys_to_str
import pickle
class analyst:
    '''
    分析类，用于执行下游分析任务，包括层次标记物发现、动态标记物发现以及通路和药物扰动分析。

    参数
    ----------------
    data_path: str
        数据文件路径（h5ad格式，例如dataset.h5ad）
    iteration: int
        用于分析的迭代次数
    target_dir: str
        结果保存目录，默认为None
    customized_drug: str
        自定义药物扰动列表，默认为None
    cmap_dir: str
        cmap数据库目录，默认为None
    customized_mode: bool
        是否为自定义模式，默认为False
    '''

    def __init__(self, data_path, iteration, target_dir=None, customized_drug=None, cmap_dir=None,
                 customized_mode=False,model_path =None):
        # 读取数据文件
        self.adata = sc.read(data_path)
        # 获取数据文件夹路径
        self.data_folder = os.path.dirname(data_path)
        # 加载属性文件
        self.adata.uns = pickle.load(open(self.data_folder + '/attribute.pkl', 'rb'))
        print(self.adata)
        # 计算疾病阶段总数
        self.total_stage = len(self.adata.obs['stage'].unique())
        # 设置自定义药物和cmap目录
        self.customized_drug = customized_drug
        self.cmap_dir = cmap_dir
        self.iteration = iteration
        self.model_path = model_path
        # 设置目标目录
        if target_dir is None:
            self.target_dir = './' + self.data_folder.split('/')[-3] + '_' + str(self.iteration)
            initalcommand = 'mkdir ' + self.target_dir
            p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)
        else:
            self.target_dir = target_dir

        # 加载训练参数
        self.model_name = f"{self.iteration-1}-model-2.pt"
        self.model_path = os.path.join(self.model_path, self.model_name)
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"找不到模型文件: {self.model_path}")
        # if customized_mode:
        #     train_params = json.load(
        #         open(os.path.join(os.path.dirname(data_path), 'model_save/training_parameters.json'), 'r'))
        # else:
        #     train_params = json.load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(data_path))),
        #                                                'model_save/training_parameters.json'), 'r'))
        # self.model_name = train_params['task'] + '_' + str(self.iteration) + '.pth'

    def perturbation_analyse_customized_pathway(self, customized_pathway, perturbed_tracks='all',
                                                overall_perturbation_analysis=True,
                                                bound=0.5, save_csv=None, save_adata=None, CUDA=False, device='cpu',
                                                random_genes=5, random_times=100):
        '''
        对自定义通路执行扰动分析

        参数
        ----------------
        customized_pathway: str
            自定义通路配置文件目录（npy文件）
        perturbed_tracks: str
            要执行扰动的轨迹，如果为'all'，则使用所有轨迹
        overall_perturbation_analysis: bool
            是否为所有轨迹计算扰动分数。如果为False，则为每个轨迹单独计算
        bound: float
            扰动后的基因表达变化阈值
        save_csv: str
            保存扰动结果的目录
        save_adata: str
            保存扰动结果的目录
        CUDA: bool
            是否使用GPU进行扰动分析
        device: str
            执行扰动分析的设备
        random_genes: int
            随机扰动的基因数量
        random_times: int
            构建随机扰动分数分布的次数
        '''
        # 计算数据与通路的重叠基因
        self.adata = calculateDataPathwayOverlapGene(self.adata, customized_pathway=customized_pathway)
        print('calculateDataPathwayOverlapGene done')
        # 计算顶级通路基因排名
        self.adata = calculateTopPathwayGeneRanking(self.adata)
        print('Start perturbation....')
        gc.collect()
        # 创建扰动对象
        a = perturbation(self.adata, self.target_dir + '/model_save/' + self.model_name, self.target_dir + '/idrem')
        # 执行通路扰动
        a.run('pathway', bound, inplace=True, CUDA=CUDA, device=device)
        # 执行随机背景扰动
        a.run('random_background', bound, inplace=True, CUDA=CUDA, device=device, random_genes=random_genes,
              random_times=random_times)
        print('random background done')
        # 分析通路扰动结果
        a.analysis('pathway', bound, perturbed_tracks, overall_perturbation_analysis)
        print('Finish results analysis')
        # 保存结果
        if save_csv is not None:
            a.uns['pathway_perturbation'].to_csv(save_csv)
        if save_adata is not None:
            a.adata.write(save_adata, compression='gzip', compression_opts=9)

    def perturbation_analyse_customized_drug(self, customized_drug, perturbed_tracks='all',
                                             overall_perturbation_analysis=True,
                                             bound=0.5, save_csv=None, save_adata=None, CUDA=True, device='cuda:0',
                                             random_genes=2, random_times=100):
        '''
        对自定义药物执行扰动分析

        参数
        ----------------
        customized_drug: str
            自定义药物配置文件目录（npy文件）
        perturbed_tracks: str
            要执行扰动的轨迹，如果为'all'，则使用所有轨迹
        overall_perturbation_analysis: bool
            是否为所有轨迹计算扰动分数。如果为False，则为每个轨迹单独计算
        bound: float
            扰动后的基因表达变化阈值
        save_csv: str
            保存扰动结果的目录
        save_adata: str
            保存扰动结果的目录
        CUDA: bool
            是否使用GPU进行扰动分析
        device: str
            执行扰动分析的设备
        random_genes: int
            随机扰动的基因数量
        random_times: int
            构建随机扰动分数分布的次数
        '''
        # 处理自定义药物数据库
        self.adata = process_customized_drug_database(self.adata, customized_drug=customized_drug)
        print('Start perturbation....')
        gc.collect()
        # 创建扰动对象
        a = perturbation(self.adata, self.target_dir + '/model_save/' + self.model_name, self.target_dir + '/idrem')
        # 执行药物扰动
        a.run('drug', bound, inplace=True, CUDA=CUDA, device=device)
        print('drug perturabtion done')
        # 执行随机背景扰动
        a.run('random_background', bound, inplace=True, CUDA=CUDA, device=device, random_genes=random_genes,
              random_times=random_times)
        print('random background done')
        # 分析药物扰动结果
        a.analysis('drug', bound, perturbed_tracks, overall_perturbation_analysis=overall_perturbation_analysis)
        print('Finish results analysis')
        # 保存结果
        if save_csv is not None:
            a.uns['drug_perturbation'].to_csv(save_csv)
        if save_adata is not None:
            a.adata.write(save_adata, compression='gzip', compression_opts=9)

    def get_median_random_gene(self, data_drug_overlapped):
        '''
        获取每个药物的随机基因数量的中位数，用于执行随机扰动

        参数
        ----------------
        data_drug_overlapped: dict
            药物与重叠基因的字典
        '''
        each_drug_data_overlappings = []
        for each in data_drug_overlapped.keys():
            each_drug_data_overlappings.append(len(data_drug_overlapped[each]))
        return np.median(each_drug_data_overlappings)

    def cmap_overlapped_genes(self, data_drug_overlapped):
        '''
        从药物重叠基因中提取基因名称（去除方向标记）
        '''
        output = []
        for each in data_drug_overlapped.keys():
            for each_gene in data_drug_overlapped[each]:
                output.append(each_gene.split(':')[0])
            break
        return output

    
    def start_analyse(self, progressionmarker_background_sampling, run_pertubration, random_times,
                      ignore_dynamic_markers=False, ignore_hcmarkers=False, customized_pathway=None,
                      defulat_perturb_change=0.5, overall_perturbation_analysis=True, perturbed_tracks='all',
                      ignore_pathway_perturabtion=False, ignore_drug_perturabtion=False, centroid=False,save_dir=None,data_path=None):
        '''
        执行下游任务，包括动态标记物发现、层次标记物发现、通路扰动和化合物扰动

        参数
        ----------------
        progressionmarker_background_sampling: int
            动态标记物发现的背景细胞采样次数
        run_pertubration: bool
            是否执行扰动分析
        random_times: int
            随机扰动次数
        ignore_dynamic_markers: bool
            是否忽略动态标记物发现，默认为False
        ignore_hcmarkers: bool
            是否忽略层次标记物发现，默认为False
        customized_pathway: str
            自定义通路，默认为None
        defulat_perturb_change: float
            扰动后的基因表达变化阈值，默认为0.5
        overall_perturbation_analysis: bool
            是否为所有轨迹进行扰动分析，默认为True
        perturbed_tracks: str
            要扰动的轨迹，默认为'all'
        ignore_pathway_perturabtion: bool
            是否忽略通路扰动，默认为False
        ignore_drug_perturabtion: bool
            是否忽略药物扰动，默认为False
        centroid: bool
            是否使用质心扰动，默认为False
        '''

        # 层次标记物发现
        if not ignore_hcmarkers:
            print('calculate hierarchical markers.....')
            hcmarkers = get_dataset_hcmarkers(self.adata, stage_key='stage', cluster_key='leiden', use_rep='umaps')
            self.adata.uns['hcmarkers'] = hcmarkers
            print('hierarchical static markers done')
            
        with open(os.path.join(data_path,'hcmarkers.pkl'), 'wb') as f:
            pickle.dump(hcmarkers, f)
        print('calculate hierarchical markers complete!')    
        # 计算数据与通路的重叠基因
        if customized_pathway is not None:
            self.adata = calculateDataPathwayOverlapGene(self.adata, customized_pathway=customized_pathway,data_path = data_path)
        else:
            self.adata = calculateDataPathwayOverlapGene(self.adata,data_path = data_path)
        print('calculateDataPathwayOverlapGene done')

        # 计算顶级通路基因排名
        self.adata = calculateTopPathwayGeneRanking(self.adata)
        print('calculateTopPathwayGeneRanking done')

        # 复制idrem结果
        if not os.path.exists(os.path.join(self.target_dir, 'idrem')):
            initalcommand = 'cp -r ' + os.path.join(os.path.dirname(self.data_folder),
                                                    'idremResults') + ' ' + self.target_dir + '/idrem'
            p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)

        # 复制模型文件
        initalcommand = 'mkdir ' + self.target_dir + '/model_save' + '&& cp ' + os.path.join(
            os.path.dirname(os.path.dirname(self.data_folder)), 'model_save',
            self.model_name) + ' ' + self.target_dir + '/model_save/' + self.model_name + '&& cp ' + os.path.join(
            os.path.dirname(os.path.dirname(self.data_folder)),
            'model_save/training_parameters.json') + ' ' + self.target_dir + '/model_save/training_parameters.json'
        p = subprocess.Popen(initalcommand, stdout=subprocess.PIPE, shell=True)

        # 动态标记物发现
        if not ignore_dynamic_markers:
            # 加载或计算进展标记物背景
            if os.path.exists(os.path.join(self.target_dir, str(
                    progressionmarker_background_sampling) + 'progressionmarker_background.npy')):
                progressionmarker_background = np.load(os.path.join(self.target_dir, str(
                    progressionmarker_background_sampling) + 'progressionmarker_background.npy'), allow_pickle=True)
                progressionmarker_background = dict(progressionmarker_background.tolist())
            else:
                progressionmarker_background = get_progressionmarker_background(
                    times=progressionmarker_background_sampling, adata=self.adata, total_stage=self.total_stage)
                np.save(os.path.join(self.target_dir,
                                     str(progressionmarker_background_sampling) + 'progressionmarker_background.npy'),
                        progressionmarker_background)

            temp_var = runGetProgressionMarker_one_dist(
                os.path.join(os.path.dirname(self.data_folder), 'idremResults'),
                progressionmarker_background,
                self.adata.shape[1],
                cutoff=0.05
            )
            print("Dynamic marker is : ",temp_var)
            # 运行进展标记物发现
            # self.adata.uns['progressionMarkers'] = temp_var
            with open(os.path.join(data_path,'dynamic_markers.pkl'), 'wb') as f:
                pickle.dump(temp_var, f)
            print('Dynamic markers discovery.....done....')
        #self.adata.uns = convert_uns_keys_to_str(self.adata.uns)
        # self.adata.write(save_dir+ '/dataset.h5ad', compression='gzip', compression_opts=9)

        # # 创建扰动运行器
        # if not centroid:
        #     perturbation_runner = perturbation(self.adata, self.target_dir + '/model_save/' + self.model_name,
        #                                        self.target_dir + '/idrem')
        # else:
        #     perturbation_runner = perturbation_centroid(self.adata, self.target_dir + '/model_save/' + self.model_name,
        #                                                 self.target_dir + '/idrem')
        #
        # # 执行扰动分析
        # if run_pertubration:
        #     direction_flag = False
        #     # 药物扰动分析准备
        #     if not ignore_drug_perturabtion:
        #         try:
        #             temp_drug = np.load(self.customized_drug, allow_pickle=True).item()
        #             if ':' in temp_drug[list(temp_drug.keys())[0]][0]:
        #                 direction_flag = True
        #         except:
        #             pass
        #
        #         if direction_flag:
        #             customized_direction = self.customized_drug
        #
        #         # 处理药物重叠和方向分配
        #         if self.customized_drug is not None:
        #             self.adata = find_overlap_and_assign_direction(self.adata, customized_drug=self.customized_drug,
        #                                                            customized_direction=customized_direction,data_path=data_path)
        #         else:
        #             if self.cmap_dir is not None:
        #                 self.adata = find_overlap_and_assign_direction(self.adata, cmap_dir=self.cmap_dir,,data_path=data_path)
        #             else:
        #                 raise ValueError('Please provide a cmap_dir or a customized drug database.')
        #
        #     print('Start perturbation....')
        #     gc.collect()
        #
        #     # 通路扰动分析
        #     if not ignore_pathway_perturabtion:
        #         perturbation_runner.run('pathway', defulat_perturb_change, inplace=True, CUDA=True)
        #         print('pathway perturbatnion done')
        #         print('building random background for pathways....')
        #         perturbation_runner.run('random_pathway_background', defulat_perturb_change, inplace=True, CUDA=True)
        #         perturbation_runner.analysis('pathway', defulat_perturb_change, perturbed_tracks,
        #                                      overall_perturbation_analysis)
        #     else:
        #         print('Ignore pathway perturbation!')
        #
        #     # 药物扰动分析
        #     if not ignore_drug_perturabtion:
        #         perturbation_runner.run('drug', defulat_perturb_change, inplace=True)
        #         print('drug perturabtion done')
        #
        #         # 随机背景扰动
        #         if self.customized_drug is not None:
        #             perturbation_runner.run('random_drug_background', defulat_perturb_change, inplace=True,
        #                                     random_times=random_times,
        #                                     random_genes=self.get_median_random_gene(
        #                                         perturbation_runner.adata.uns['data_drug_overlap_genes']))
        #         else:
        #             random_gene = self.cmap_overlapped_genes(perturbation_runner.adata.uns['data_drug_overlap_genes'])
        #             random_genes = []
        #             import random
        #
        #             # 生成随机基因列表
        #             for time in range(progressionmarker_background_sampling):
        #                 random.seed(time)
        #                 choices = [random.choice(['+', '-']) for _ in range(len(random_gene))]
        #                 random_genes.append([random_gene[i] + ':' + choices[i] for i in range(len(random_gene))])
        #
        #             perturbation_runner.adata.uns['cmap_random_genes'] = random_genes
        #             perturbation_runner.run('random_drug_background', defulat_perturb_change, inplace=True,
        #                                     random_genes=random_genes)
        #
        #         # 分析药物扰动结果
        #         perturbation_runner.analysis('drug', defulat_perturb_change, perturbed_tracks,
        #                                      overall_perturbation_analysis)
        #         print('analysis of drug perturbation')
        #     else:
        #         print('Ignore drug perturbation!')

        # 保存结果
        # perturbation_runner.adata.uns['hcmarkers'] = hcmarkers
        # with open(os.path.join(self.target_dir, 'attribute.pkl'), 'wb') as f:
        #     pickle.dump(perturbation_runner.adata.uns, f)
        #
        # del perturbation_runner.adata.uns
        # # 转换数据类型
        # perturbation_runner.adata.obs['leiden'] = perturbation_runner.adata.obs['leiden'].astype(str)
        # perturbation_runner.adata.obs['stage'] = perturbation_runner.adata.obs['stage'].astype(str)
        # perturbation_runner.adata.obs['ident'] = perturbation_runner.adata.obs['ident'].astype(str)
        # # 保存最终结果
        # perturbation_runner.adata.write(self.target_dir + '/dataset.h5ad', compression='gzip', compression_opts=9)