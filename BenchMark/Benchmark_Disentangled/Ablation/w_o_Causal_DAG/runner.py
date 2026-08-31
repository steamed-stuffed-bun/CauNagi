import subprocess
# from .utils.gcn_utils import setup_graph
# from .utils.CPO_utils import get_neighbors, auto_resolution
# from .utils.attribute_utils import saveRep, get_all_adj_adata, mergeAdata, updateAttributes, get_data_file_path
# from .dynamic_graphs.buildGraph import getandUpadateEdges
import gc
import scanpy as sc
import numpy as np
import os
from scipy.sparse import csr_matrix, issparse
from .attribute_utils import updateAttributes, get_all_adj_adata, get_data_file_path, mergeAdata
from .CPO_utils import get_neighbors, auto_resolution
from .processIDREM import getClusterPaths, getClusterIdrem, runIdrem
from .processTFs import getTFs, getTargetGenes, matchTFandTGWithFoldChange, updataGeneTablesWithDecay
#     updataGeneTablesWithDecay
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from .buildGraph import getandUpadateEdges


class Caunagi_runner:

    def __init__(self, data_path, temp_path, total_stage,iteration, trainer, idrem_dir, concept_list, concept_cdag,
                 concept_counts, training_profile_size,CellType_dicts,
                 adversarial=True,
                 connect_edges_cutoff=0.05):
        self.origin_data_path = data_path
        self.data_path = temp_path
        self.total_stage = total_stage
        self.iteration = iteration
        self.trainer = trainer
        self.resolutions = None  # 存储各阶段聚类分辨率
        self.idrem_dir = idrem_dir
        self.concept_list = concept_list
        self.concept_cdag = concept_cdag
        self.concept_counts = concept_counts
        self.profile_size = training_profile_size
        self.CellType_dicts = CellType_dicts
        self.neighbor_parameters = None  # 存储各阶段的邻居数量参数
        self.setup_CPO = False  # 是否已设置聚类参数优化(CPO)的标志
        self.species = None  # 物种信息（人类/小鼠）
        self.setup_IDREM = False  # 是否已设置iDREM参数的标志
        self.adversarial = adversarial
        self.connect_edges_cutoff = connect_edges_cutoff

    def load_stage_data(self):
        '''
        从data_path加载各阶段数据，并创建汇总数据集
        '''
        stageadata = []
        for i in range(self.total_stage):
            CellType_dicts = self.CellType_dicts
            if self.iteration == 0:
                adata = sc.read_h5ad(os.path.join(self.origin_data_path, '%d.h5ad'% (i)))
                
                celltype_list = list(adata.obs['CellType'])
                adata.obs['celltype'] = celltype_list
                simple_list = [CellType_dicts[i] for i in list(adata.obs['CellType'])]
                adata.obs['CellType'] = simple_list
            else:
                adata = sc.read_h5ad(os.path.join(self.data_path, '%d/stagedata'%(self.iteration-1)+'/%d.h5ad'% (i)))

            
            if issparse(adata.X):
                adata.X = adata.X.toarray()
            
            #simple_list = [CellType_dicts[i] for i in list(adata.obs['name.simple'])]
            #adata.obs['name.simple'] = simple_list
          
            # 清理旧聚类结果
            if 'leiden_colors' in adata.uns:
                del adata.uns['leiden_colors']
            stageadata.append(adata)
        # 创建合并所有阶段的邻接数据集
        # self.all_in_one = get_all_adj_adata(stageadata)
        self.adata_stages = stageadata
        self.genenames = np.array(list(self.adata_stages[0].var.index.values))  # 保存基因名称

    def annotate_stage_data(self, adata, stage, CPO):
        # 获取潜在表示
        z_locs, z_scales, cell_embeddings = self.trainer.get_latent_representation(adata)
        adata.obsm['z'] = cell_embeddings

        # 如果未设置邻居参数，使用默认值
        if self.neighbor_parameters is None:
            sc.pp.neighbors(adata, use_rep="z", n_neighbors=50, method='umap')
            return adata

        # 清空现有邻居信息
        if 'connectivities' in adata.obsp.keys():
            del adata.obsp['connectivities']
            del adata.obsp['distances']

        # 计算k近邻图
        z_adj = kneighbors_graph(adata.obsm['z'], self.neighbor_parameters[stage], mode='connectivity',
                                 include_self=True, n_jobs=20)
        adata.obsp['connectivities'] = z_adj
        adata.obsp['distances'] = kneighbors_graph(adata.obsm['z'], self.neighbor_parameters[stage], mode='distance',
                                                   include_self=True, n_jobs=20)

        # 如果不是CPO模式，重新计算邻居图
        if CPO is False:
            sc.pp.neighbors(adata, use_rep="z", n_neighbors=self.neighbor_parameters[stage], method='umap')

        # 执行Leiden聚类
        sc.tl.leiden(adata, resolution=self.resolutions[stage])

        # 执行PAGA分析
        sc.tl.paga(adata)
        sc.pl.paga(adata, show=False)

        # 基于PAGA初始化UMAP
        sc.tl.umap(adata, min_dist=0.05, init_pos='paga')

        rep = [z_locs, z_scales]
        adata.obs['ident'] = 'None'
        adata.obs['leiden'] = adata.obs['leiden'].astype(str)

        # 可视化聚类结果
        sc.pl.umap(adata, color='leiden', show=False)

        # 更新属性：添加顶部基因、细胞类型注释等
        adata, averageValue, reps = updateAttributes(adata, rep)

        adata.obs['leiden'] = adata.obs['leiden'].astype(str)

        # 初始化基因权重层
        # if self.iteration == 0:
        #     allzeros = np.zeros_like(adata.X)
        #     allzeros = csr_matrix(allzeros)
        #     adata.layers['geneWeight'] = allzeros
        allzeros = np.zeros_like(adata.X)
        allzeros = csr_matrix(allzeros)
        adata.layers['geneWeight'] = allzeros

        # 可视化细胞标识
        sc.pl.umap(adata, color='ident', show=False)

        # 保存结果
        # adata.write(os.path.join(self.data_path, str(self.iteration) + '/stagedata/%d.h5ad' % stage),
        #             compression='gzip', compression_opts=9)

        adata.write(os.path.join(self.data_path, str(self.iteration) + '/stagedata/%d.h5ad' % stage),
                    compression='gzip', compression_opts=9)

        return adata, averageValue, reps

    def set_up_CPO(self, anchor_neighbors, max_neighbors, min_neighbors, resolution_min, resolution_max):
        '''
        设置聚类参数优化(CPO)的参数

        参数
        ------------
        anchor_neighbors: int
            锚点细胞的邻居数量
        max_neighbors: int
            最大邻居数
        min_neighbors: int
            最小邻居数
        resolution_min: float
            最小分辨率
        resolution_max: float
            最大分辨率
        '''
        self.setup_CPO = True
        self.anchor_neighbors = anchor_neighbors
        self.max_neighbors = max_neighbors
        self.min_neighbors = min_neighbors
        self.resolution_min = resolution_min
        self.resolution_max = resolution_max

    def run_CPO(self):
        '''
        执行聚类参数优化(CPO)
        '''
        # 计算各阶段细胞数
        max_adata_cells = 0
        num_cells = []
        for each in self.adata_stages:
            num_cells.append(each.shape[0])
            if len(each) > max_adata_cells:
                max_adata_cells = len(each)

        self.resolution_coefficient = max_adata_cells

        # 暂时注释数据以进行参数优化
        for i in range(0, len(self.adata_stages)):
            self.adata_stages[i] = self.annotate_stage_data(self.adata_stages[i], i, CPO=True)

        # 使用默认参数如果未设置
        if not self.setup_CPO:
            print('CPO参数未设置，使用默认参数')
            print(
                'anchor_neighbors: 15, max_neighbors: 35, min_neighbors: 10, resolution_min: 0.8, resolution_max: 1.5')
            self.neighbor_parameters, anchor_index = get_neighbors(self.adata_stages, num_cells, anchor_neighbors=15,
                                                                   max_neighbors=35, min_neighbors=10)
            self.resolutions, _ = auto_resolution(self.adata_stages, anchor_index, self.neighbor_parameters, 0.8, 1.5)
        else:
            # 使用自定义参数
            self.neighbor_parameters, anchor_index = get_neighbors(self.adata_stages, num_cells,
                                                                   anchor_neighbors=self.anchor_neighbors,
                                                                   max_neighbors=self.max_neighbors,
                                                                   min_neighbors=self.min_neighbors)
            self.resolutions, _ = auto_resolution(self.adata_stages, anchor_index, self.neighbor_parameters,
                                                  self.resolution_min, self.resolution_max)

    def update_cell_attributes(self, CPO):
        '''
        更新和保存细胞属性（顶部基因、细胞类型、潜在表示）
        '''
        self.averageValues = []  # 各聚类的平均表达值
        reps = []  # 潜在表示

        for i in range(0, len(self.adata_stages)):
            adata = self.adata_stages[i]
            # 初始化存储空间
            adata.uns['topGene'] = {}
            adata.uns['clusterType'] = {}
            adata.uns['rep'] = {}

            # 注释当前阶段数据
            adata, averageValue, rep = self.annotate_stage_data(adata, i, CPO)

            self.adata = adata
            gc.collect()  # 垃圾回收

            reps.append(rep)
            averageValue = np.array(averageValue)
            self.averageValues.append(averageValue)

        # 保存平均表达值
        self.averageValues = np.array(self.averageValues, dtype=object)
        np.save(os.path.join(self.data_path, '%d/averageValues.npy'%(self.iteration)), self.averageValues)
        # 保存潜在表示
        np.save(os.path.join(self.data_path, '%d/rep.npy'%(self.iteration)), np.array(reps, dtype=object))

    def build_temporal_dynamics_graph(self):
        '''
        构建时序动态图
        '''
        self.edges = getandUpadateEdges(
            self.total_stage,
            self.data_path,
            self.iteration,
            self.connect_edges_cutoff
        )

    def set_up_IDREM(self, Minimum_Absolute_Log_Ratio_Expression, Convergence_Likelihood, Minimum_Standard_Deviation):
        '''
        设置iDREM运行参数

        参数
        ------------
        Minimum_Absolute_Log_Ratio_Expression: float
            最小绝对对数比率表达量
        Convergence_Likelihood: float
            收敛似然值
        Minimum_Standard_Deviation: float
            最小标准差
        '''
        self.setup_IDREM = True
        self.Minimum_Absolute_Log_Ratio_Expression = Minimum_Absolute_Log_Ratio_Expression
        self.Convergence_Likelihood = Convergence_Likelihood
        self.Minimum_Standard_Deviation = Minimum_Standard_Deviation

    def set_up_species(self, species):
        '''
        设置物种信息

        参数
        ------------
        species: str
            物种（'Human' 或 'Mouse'）
        '''
        print('物种: 使用%s数据' % species)
        self.species = species

    def run_IDREM(self):
        '''
        运行iDREM软件
        '''
        # 加载平均表达值
        averageValues = np.load(
            os.path.join(self.data_path, '%d/averageValues.npy' % self.iteration),
            allow_pickle=True
        )

        # 获取聚类路径（完整时序路径）
        paths = getClusterPaths(self.edges, self.total_stage)
        idrem = getClusterIdrem(paths, averageValues, self.total_stage)
        paths = [each for each in paths.values() if len(each) == self.total_stage]  # 筛选完整路径

        idrem = np.array(idrem)
        self.genenames = np.array(list(self.adata_stages[0].var.index.values))

        # 检查参数是否设置
        if not self.setup_IDREM:
            print('IDREM参数未设置，使用默认参数')
            print(
                'Minimum_Absolute_Log_Ratio_Expression: 0.5, Convergence_Likelihood: 0.001, Minimum_Standard_Deviation: 0.5')

            if self.species is None:
                print('默认使用人类数据')
                runIdrem(paths, self.data_path, idrem, self.genenames, self.iteration, self.idrem_dir)
            else:
                runIdrem(paths, self.data_path, idrem, self.genenames, self.iteration, self.idrem_dir,
                         species=self.species)
        else:
            if self.species is None:
                print('默认使用人类数据')
                runIdrem(paths, self.data_path, idrem, self.genenames, self.iteration, self.idrem_dir,
                         Minimum_Absolute_Log_Ratio_Expression=self.Minimum_Absolute_Log_Ratio_Expression,
                         Convergence_Likelihood=self.Convergence_Likelihood,
                         Minimum_Standard_Deviation=self.Minimum_Standard_Deviation)
            else:
                runIdrem(paths, self.data_path, idrem, self.genenames, self.iteration, self.idrem_dir,
                         species=self.species,
                         Minimum_Absolute_Log_Ratio_Expression=self.Minimum_Absolute_Log_Ratio_Expression,
                         Convergence_Likelihood=self.Convergence_Likelihood,
                         Minimum_Standard_Deviation=self.Minimum_Standard_Deviation)

    def update_gene_weights_table(self, topN=100):
        '''
        更新基因权重表

        参数
        ------------
        topN: int, 可选 (默认=100)
            选择的顶部基因数量
        '''
        # 获取转录因子(TFs)
        TFs = getTFs(
            os.path.join(self.data_path, str(self.iteration) + '/idremResults/'),
            total_stage=self.total_stage
        )

        # 获取目标基因
        scope = getTargetGenes(
            os.path.join(self.data_path, str(self.iteration) + '/idremResults/'),
            topN
        )

        # 加载平均表达值
        self.averageValues = np.load(
            os.path.join(self.data_path, '%d/averageValues.npy' % self.iteration),
            allow_pickle=True
        )

        # 根据物种匹配TF和目标基因
        if self.species == 'Human':
            p = matchTFandTGWithFoldChange(
                TFs, scope, self.averageValues,
                get_data_file_path(self.origin_data_path,'human_encode.txt'),
                self.genenames, self.total_stage
            )
        elif self.species == 'Mouse':
            p = matchTFandTGWithFoldChange(
                TFs, scope, self.averageValues,
                get_data_file_path(self.origin_data_path,'mouse_predicted.txt'),
                self.genenames, self.total_stage
            )

        # 使用衰减更新基因表
        updateLoss = updataGeneTablesWithDecay(
            self.data_path,
            str(self.iteration),
            p,
            self.total_stage
        )

    def build_iteration_dataset(self):
        '''
        构建迭代数据集
        '''
        mergeAdata(
            os.path.join(self.data_path, str(self.iteration)),
            total_stages=self.total_stage
        )

    def run(self, CPO):

        self.load_stage_data()
        # 训练模型（注意：is_iterative可能未定义，原代码如此）
        self.trainer.train()
        self.trainer.load_trained(
                                  self.concept_list,
                                  self.concept_counts,
                                  self.concept_cdag
                                  )
        # 聚类参数优化或使用默认值
        if CPO:
            self.run_CPO()
        else:
            self.resolutions = [1.0] * self.total_stage
            self.neighbor_parameters = [30] * self.total_stage

        # 更新细胞属性
        self.update_cell_attributes(CPO)

        # 构建时序图
        self.build_temporal_dynamics_graph()
        # 运行iDREM
        self.run_IDREM()
        # 更新基因权重
        self.update_gene_weights_table()
        # 构建迭代数据集
        self.build_iteration_dataset()
