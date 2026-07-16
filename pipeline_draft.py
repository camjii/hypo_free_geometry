"""
This pipeline does the following:

1. Extracts model activations for a given feature
2. Constructs a distance matrix
3. Build an epsilon neighborhood graph, which is essentially takes the distance between each point in the distance matrix, and draws a connection between them if d(Xi, Xj) <= ε (Sk_learn, radius_neighbors_graph)
4. Runs the graph through PH (ripser.py, giotto-tda, VietorisRipsPersistence), builds on it with higher simplicial complexes, and gets a persistence diagram, showing what features upheld through all tests, and where structures dissipated
5. Get the curvature signature of the ε-graph via ollivier ricci curvature (GraphRicciCurvature)
"""

import os

# networkit (used by GraphRicciCurvature for all-pairs-shortest-path) bundles
# its own OpenMP runtime; with torch/scipy/sklearn each bundling their own too,
# multiple OpenMP runtimes threading in one process segfaults on macOS. Forcing
# single-threaded OpenMP here sidesteps the conflict. Must be set before any
# native lib below is imported -- OpenMP reads it at library init time.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

# import transformer_lens
from transformer_lens import HookedTransformer
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from ripser import ripser
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
# import persim
from sklearn.neighbors import radius_neighbors_graph
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skdim

from null_cloud import Manifold
from topology_metric import TopologyMetric


class Pipeline():
    def __init__(self,pos_prompts):
        self.model = HookedTransformer.from_pretrained('gpt2')
        self.pos_prompts = pos_prompts #get from CKA github
        

    
    
    def select_layer_by_topology(self, kind='noise', n_null=3, max_dim=1):
        '''
        Layer selection by shape, not magnitude: score each layer by how far its
        activation cloud sits from its own null.
        '''
        activations_dict = {f'layer_{l + 1}':[] for l in range(self.model.cfg.n_layers)}

        for prompt in self.pos_prompts:
            with torch.no_grad():
                _, cache = self.model.run_with_cache(prompt)
                for l in range(self.model.cfg.n_layers):
                    activations_dict[f'layer_{l+1}'].append(cache['resid_post', l][0, -1, :])

        for layer, _ in activations_dict.items():
            activations_dict[layer] = torch.stack(activations_dict[layer]).detach().cpu().numpy()

        best_layer, best_score, scores = None, -np.inf, {}
        for layer, act in activations_dict.items():
            manifold = Manifold(self, act, np.zeros_like(act), cloud=act, label=layer)
            tm = TopologyMetric(manifold, kind=kind, n_null=n_null, max_dim=max_dim)
            scores[layer] = tm.metric

            if tm.topology > best_score:
                best_score, best_layer = tm.topology, layer

        print(f'Layer with the strongest topology-vs-null signal is {best_layer} (topology={best_score:.4f})')

        return best_layer, activations_dict[best_layer], scores

    
    '''
    Replace the create_distance_matrix function with running PCA on contrastive_diff

    Figure out how a persistence diagram can be created from PCA
    '''
    
    
    def reduce_pca(self, contrastive_diff, var_threshold=0.95):
        #Analysis reduction: keep enough components to capture the concept.
        X = contrastive_diff.detach().cpu().numpy() if isinstance(contrastive_diff, torch.Tensor) else contrastive_diff
        pca = PCA(n_components=min(X.shape))
        full = pca.fit_transform(X)
        m = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold)) + 1
        print(f'PCA: keeping m={m} components ({var_threshold*100:.0f}% variance)')
        return full[:, :m]                            # m vectors fed into ripser/curvature

    def plot_pca(self, opt_pos_activations):
        #Visualization only: 2-D projection
        X = opt_pos_activations.detach().cpu().numpy() if isinstance(opt_pos_activations, torch.Tensor) else opt_pos_activations
        projected = PCA(n_components=2).fit_transform(X)

        plt.figure()
        plt.scatter(projected[:, 0], projected[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of layer activations')
        plt.show()

        return projected

    def get_intrinsic_dim(self, contrastive_diff):
        X = contrastive_diff.numpy() if isinstance(contrastive_diff, torch.Tensor) else contrastive_diff
        d = skdim.id.TwoNN().fit(X).dimension_
        print(f'Intrinsic dimension (TwoNN): {d:.2f}')
        return d #returns ID using TwoNN

    def create_persistence_diagram(self, projected):   #persistent homology from (eps = 1 to inf)
        persist_diagram = ripser(projected, maxdim = 1, distance_matrix=False, do_cocycles = False, n_perm = None )
        return persist_diagram
    
    
    def create_persistence_vector(self,persist_diagram):
        #use persim.PersistenceImager
        pass
    
    
    def create_epsilon_graph(self,projected, eps):
        graph = radius_neighbors_graph(projected,radius = eps,mode = 'distance', metric='euclidean') #nxn matrix of weights that connect edges
        '''
        mode = 'distance' ensures that the graph is not binary (when all distances are 1.0)
        '''
        
        graph = nx.Graph(graph) 
        
        return graph 
   
    def compute_ollivier_ricci(self, graph):
        orc = OllivierRicci(graph, alpha = 0.5, proc = 1, verbose = 'ERROR')
        orc_curv = orc.compute_ricci_curvature()

        raw_values = []
        for edge in orc_curv.edges(data='ricciCurvature'):
            raw_values.append(edge[-1]) #structure for each edge between points (u,v) is (u,v,curvature_value)
        
        mean_curv = np.mean(raw_values) 

        summ_dict = {'graph': orc_curv, 'mean_curvature': mean_curv, 'raw_values':raw_values}

        return summ_dict
    
   

