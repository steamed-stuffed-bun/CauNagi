import scanpy as sc
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import issparse


def _sorted_neighbor_distances(adata, max_neighbors):
    """Return a dense, padded matrix of nearest-neighbor distances."""
    distances = adata.obsp.get('distances')
    if distances is None:
        raise ValueError("Stage data must contain a distances graph before CPO.")
    distances = distances.toarray() if issparse(distances) else np.asarray(distances)
    distances = np.sort(distances, axis=1)
    distances = distances[:, 1:] if distances.shape[1] else distances
    distances = distances[:, :max_neighbors]
    if distances.shape[1] == 0:
        return np.zeros((adata.n_obs, 1))
    if distances.shape[1] < max_neighbors:
        distances = np.pad(
            distances,
            ((0, 0), (0, max_neighbors - distances.shape[1])),
            mode='edge',
        )
    return distances


def get_neighbors(stagedata, num_cells, anchor_neighbors, max_neighbors, min_neighbors):
    if not stagedata:
        raise ValueError("At least one stage is required for CPO.")
    max_neighbors = max(1, int(max_neighbors))
    min_neighbors = max(1, int(min_neighbors))
    anchor_neighbors = max(1, int(anchor_neighbors))
    temp_num_cells = num_cells.copy()
    temp_num_cells.sort()
    median_index = int(len(temp_num_cells) / 2)
    anchor = temp_num_cells[median_index]

    anchor_index = num_cells.index(anchor)
    temp_adata = stagedata[anchor_index]
    # distance = temp_adata.obsp['distances'].toarray()
    distance = _sorted_neighbor_distances(temp_adata, max_neighbors)

    avg_anchor_distance = np.mean(distance[:, min(anchor_neighbors, max_neighbors) - 1])
    neighbors = []
    for i in range(len(num_cells)):
        if i != anchor_index:
            temp_adata = stagedata[i]
            # distance = temp_adata.obsp['distances'].toarray()
            distance = _sorted_neighbor_distances(temp_adata, max_neighbors)
            search = []
            for neighbor in range(max_neighbors):
                search.append(abs(np.mean(distance[:, neighbor]) - avg_anchor_distance))
            # find the index of minimum value in search
            min_index = search.index(min(search)) + 1
            min_index = min(min_index, max(1, len(temp_adata) - 1))
            if min_index < min_neighbors:
                min_index = min_neighbors
            min_index = min(min_index, max(1, len(temp_adata) - 1))

            neighbors.append(min_index)
        else:
            neighbors.append(anchor_neighbors)
    return neighbors, anchor_index



def get_mean_median_cell_population(adata):
    num_cells = []
    for i in list(adata.obs['leiden'].unique()):
        temp = adata.obs[adata.obs['leiden'] == i].index.tolist()
        temp = adata[temp]
        num_cells.append(len(temp))
    return np.mean(num_cells)/len(adata), np.median(num_cells)/len(adata)



def auto_resolution(stagedata, anchor_index, neighbors, min_res, max_res):
    anchor_adata = stagedata[anchor_index]
    anchor_neighbors = min(neighbors[anchor_index], max(1, anchor_adata.n_obs - 1))
    anchor_adata.obsp['connectivities'] = kneighbors_graph(anchor_adata.obsm['z'], anchor_neighbors,
                                                           mode='connectivity', include_self=True, n_jobs=20)
    anchor_adata.obsp['distances'] = kneighbors_graph(anchor_adata.obsm['z'], anchor_neighbors, mode='distance',
                                                      include_self=True, n_jobs=20)
    sc.tl.leiden(anchor_adata, resolution=min_res)
    all_means = []
    anchor_mean, anchor_median = get_mean_median_cell_population(anchor_adata)
    out_res = []
    for i in range(len(stagedata)):
        differences = []
        temp_all_means = []
        if i != anchor_index:
            temp_adata = stagedata[i]
            stage_neighbors = min(neighbors[i], max(1, temp_adata.n_obs - 1))
            temp_adata.obsp['connectivities'] = kneighbors_graph(temp_adata.obsm['z'], stage_neighbors,
                                                                 mode='connectivity', include_self=True, n_jobs=20)
            temp_adata.obsp['distances'] = kneighbors_graph(temp_adata.obsm['z'], stage_neighbors, mode='distance',
                                                            include_self=True, n_jobs=20)
            for j in np.arange(min_res, max_res + 0.1, 0.1):
                sc.tl.leiden(temp_adata, resolution=j)
                temp_mean, temp_median = get_mean_median_cell_population(temp_adata)
                temp_all_means.append(temp_mean)
                differences.append(abs(temp_mean - anchor_mean))
            min_index = differences.index(min(differences))
            out_res.append(min_index * 0.1 + min_res)
            all_means.append(temp_all_means[min_index])
        else:
            out_res.append(1)
            all_means.append(anchor_mean)
    return out_res, all_means

