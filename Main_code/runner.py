import gc
import scanpy as sc
import numpy as np
import os
from scipy.sparse import issparse
from .attribute_utils import updateAttributes, get_data_file_path, mergeAdata
from .CPO_utils import get_neighbors, auto_resolution
from .processIDREM import getClusterPaths, getClusterIdrem, runIdrem
from .processTFs import getTFs, getTargetGenes, matchTFandTGWithFoldChange, updateGeneTablesWithDecay
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
        self.resolutions = None
        self.idrem_dir = idrem_dir
        self.concept_list = concept_list
        self.concept_cdag = concept_cdag
        self.concept_counts = concept_counts
        self.profile_size = training_profile_size
        self.CellType_dicts = CellType_dicts
        self.neighbor_parameters = None
        self.setup_CPO = False
        self.species = "Human"
        self.setup_IDREM = False
        self.adversarial = adversarial
        self.connect_edges_cutoff = connect_edges_cutoff

    def load_stage_data(self):
        '''
        Load each stage and prepare the stage-level datasets.
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
          
            if 'leiden_colors' in adata.uns:
                del adata.uns['leiden_colors']
            stageadata.append(adata)
        # self.all_in_one = get_all_adj_adata(stageadata)
        self.adata_stages = stageadata
        self.genenames = np.array(list(self.adata_stages[0].var.index.values))

    def annotate_stage_data(self, adata, stage, CPO):
        z_locs, z_scales, cell_embeddings = self.trainer.get_latent_representation(adata)
        adata.obsm['z'] = cell_embeddings

        if self.neighbor_parameters is None:
            n_neighbors = min(50, max(1, adata.n_obs - 1))
            sc.pp.neighbors(adata, use_rep="z", n_neighbors=n_neighbors, method='umap')
            return adata

        if 'connectivities' in adata.obsp.keys():
            del adata.obsp['connectivities']
            del adata.obsp['distances']

        n_neighbors = min(self.neighbor_parameters[stage], max(1, adata.n_obs - 1))
        z_adj = kneighbors_graph(adata.obsm['z'], n_neighbors, mode='connectivity',
                                  include_self=True, n_jobs=20)
        adata.obsp['connectivities'] = z_adj
        adata.obsp['distances'] = kneighbors_graph(adata.obsm['z'], n_neighbors, mode='distance',
                                                   include_self=True, n_jobs=20)

        # In non-CPO mode, rebuild the neighbor graph with the selected value.
        if CPO is False:
            sc.pp.neighbors(adata, use_rep="z", n_neighbors=n_neighbors, method='umap')

        sc.tl.leiden(adata, resolution=self.resolutions[stage])

        sc.tl.paga(adata)

        sc.tl.umap(adata, min_dist=0.05, init_pos='paga')

        rep = [z_locs, z_scales]
        adata.obs['ident'] = 'None'
        adata.obs['leiden'] = adata.obs['leiden'].astype(str)

        adata, averageValue, reps = updateAttributes(adata, rep)

        adata.obs['leiden'] = adata.obs['leiden'].astype(str)

        if 'geneWeight' not in adata.layers:
            adata.layers['geneWeight'] = np.zeros(adata.shape, dtype=np.float32)

        adata.write(os.path.join(self.data_path, str(self.iteration) + '/stagedata/%d.h5ad' % stage),
                    compression='gzip', compression_opts=9)

        return adata, averageValue, reps

    def set_up_CPO(self, anchor_neighbors, max_neighbors, min_neighbors, resolution_min, resolution_max):
        '''
        Configure clustering parameter optimization.

        Parameters
        ------------
        anchor_neighbors: int
            Number of neighbors for the anchor stage.
        max_neighbors: int
            Maximum number of neighbors.
        min_neighbors: int
            Minimum number of neighbors.
        resolution_min: float
            Minimum Leiden resolution.
        resolution_max: float
            Maximum Leiden resolution.
        '''
        self.setup_CPO = True
        self.anchor_neighbors = anchor_neighbors
        self.max_neighbors = max_neighbors
        self.min_neighbors = min_neighbors
        self.resolution_min = resolution_min
        self.resolution_max = resolution_max

    def run_CPO(self):
        '''
        Run clustering parameter optimization.
        '''
        max_adata_cells = 0
        num_cells = []
        for each in self.adata_stages:
            num_cells.append(each.shape[0])
            if len(each) > max_adata_cells:
                max_adata_cells = len(each)

        self.resolution_coefficient = max_adata_cells

        for i in range(0, len(self.adata_stages)):
            self.adata_stages[i] = self.annotate_stage_data(self.adata_stages[i], i, CPO=True)

        # Use default CPO parameters when none were registered.
        if not self.setup_CPO:
            print('CPO parameters were not registered; using defaults.')
            self.neighbor_parameters, anchor_index = get_neighbors(self.adata_stages, num_cells, anchor_neighbors=15,
                                                                   max_neighbors=35, min_neighbors=10)
            self.resolutions, _ = auto_resolution(self.adata_stages, anchor_index, self.neighbor_parameters, 0.8, 1.5)
        else:
            # Use the registered CPO parameters.
            self.neighbor_parameters, anchor_index = get_neighbors(self.adata_stages, num_cells,
                                                                   anchor_neighbors=self.anchor_neighbors,
                                                                   max_neighbors=self.max_neighbors,
                                                                   min_neighbors=self.min_neighbors)
            self.resolutions, _ = auto_resolution(self.adata_stages, anchor_index, self.neighbor_parameters,
                                                  self.resolution_min, self.resolution_max)

    def update_cell_attributes(self, CPO):
        '''
        Update and save cell attributes, marker genes, and latent representations.
        '''
        self.averageValues = []
        reps = []

        for i in range(0, len(self.adata_stages)):
            adata = self.adata_stages[i]
            adata.uns['topGene'] = {}
            adata.uns['clusterType'] = {}
            adata.uns['rep'] = {}

            adata, averageValue, rep = self.annotate_stage_data(adata, i, CPO)

            self.adata = adata
            gc.collect()

            reps.append(rep)
            averageValue = np.array(averageValue)
            self.averageValues.append(averageValue)

        # Save average cluster expression.
        self.averageValues = np.array(self.averageValues, dtype=object)
        np.save(os.path.join(self.data_path, '%d/averageValues.npy'%(self.iteration)), self.averageValues)
        # Save Gaussian cluster representations.
        np.save(os.path.join(self.data_path, '%d/rep.npy'%(self.iteration)), np.array(reps, dtype=object))

    def build_temporal_dynamics_graph(self):
        '''
        Build the temporal graph.
        '''
        self.edges = getandUpadateEdges(
            self.total_stage,
            self.data_path,
            self.iteration,
            self.connect_edges_cutoff
        )

    def set_up_IDREM(self, Minimum_Absolute_Log_Ratio_Expression, Convergence_Likelihood, Minimum_Standard_Deviation):
        '''
        Configure iDREM.

        Parameters
        ------------
        Minimum_Absolute_Log_Ratio_Expression: float
            Minimum absolute log-ratio expression.
        Convergence_Likelihood: float
            Convergence likelihood.
        Minimum_Standard_Deviation: float
            Minimum standard deviation.
        '''
        self.setup_IDREM = True
        self.Minimum_Absolute_Log_Ratio_Expression = Minimum_Absolute_Log_Ratio_Expression
        self.Convergence_Likelihood = Convergence_Likelihood
        self.Minimum_Standard_Deviation = Minimum_Standard_Deviation

    def set_up_species(self, species):
        '''
        Set the species used by the iDREM reference network.

        Parameters
        ------------
        species: str
            Species (Human or Mouse).
        '''
        print('Using %s reference data.' % species)
        self.species = species

    def run_IDREM(self):
        '''
        Run iDREM.
        '''
        averageValues = np.load(
            os.path.join(self.data_path, '%d/averageValues.npy' % self.iteration),
            allow_pickle=True
        )

        paths = getClusterPaths(self.edges, self.total_stage)
        idrem = getClusterIdrem(paths, averageValues, self.total_stage)
        paths = [each for each in paths.values() if len(each) == self.total_stage]
        if not paths:
            raise RuntimeError(
                "No complete stage-to-stage cluster paths were found. "
                "Adjust clustering parameters or the temporal-edge cutoff."
            )

        idrem = np.array(idrem)
        self.genenames = np.array(list(self.adata_stages[0].var.index.values))

        if not self.setup_IDREM:
            runIdrem(paths, self.data_path, idrem, self.genenames, self.iteration, self.idrem_dir,
                     species=self.species)
        else:
            runIdrem(paths, self.data_path, idrem, self.genenames, self.iteration, self.idrem_dir,
                     species=self.species,
                     Minimum_Absolute_Log_Ratio_Expression=self.Minimum_Absolute_Log_Ratio_Expression,
                     Convergence_Likelihood=self.Convergence_Likelihood,
                     Minimum_Standard_Deviation=self.Minimum_Standard_Deviation)

    def update_gene_weights_table(self, topN=100):
        '''
        Update the iterative gene-evidence layer.

        Parameters
        ------------
        topN: int, optional (default=100)
            Number of top genes to use.
        '''
        TFs = getTFs(
            os.path.join(self.data_path, str(self.iteration) + '/idremResults/'),
            total_stage=self.total_stage
        )

        scope = getTargetGenes(
            os.path.join(self.data_path, str(self.iteration) + '/idremResults/'),
            topN
        )

        self.averageValues = np.load(
            os.path.join(self.data_path, '%d/averageValues.npy' % self.iteration),
            allow_pickle=True
        )

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

        updateGeneTablesWithDecay(
            self.data_path,
            str(self.iteration),
            p,
            self.total_stage
        )

    def build_iteration_dataset(self):
        '''
        Merge the stage outputs into the next-iteration dataset.
        '''
        mergeAdata(
            os.path.join(self.data_path, str(self.iteration)),
            total_stages=self.total_stage
        )

    def run(self, CPO):

        self.load_stage_data()
        self.trainer.train()
        self.trainer.load_trained(self.concept_list, self.concept_counts, self.concept_cdag)
        if CPO:
            self.run_CPO()
        else:
            self.resolutions = [1.0] * self.total_stage
            self.neighbor_parameters = [30] * self.total_stage

        # Update cell attributes.
        self.update_cell_attributes(CPO)

        # Build the temporal graph.
        self.build_temporal_dynamics_graph()
        # Run iDREM.
        self.run_IDREM()
        # Update iterative gene weights.
        self.update_gene_weights_table()
        # Build the merged iteration dataset.
        self.build_iteration_dataset()
