"""
This pipeline does the following:

1. Extracts model activations for a given feature
2. Constructs a distance matrix
3. Build an epsilon neighborhood graph, which is essentially takes the distance between each point in the distance matrix, and draws a connection between them if d(Xi, Xj) <= ε (Sk_learn, radius_neighbors_graph)
4. Runs the graph through PH (ripser.py, giotto-tda, VietorisRipsPersistence), builds on it with higher simplicial complexes, and gets a persistence diagram, showing what features upheld through all tests, and where structures dissipated
5. Get the curvature signature of the ε-graph via ollivier ricci curvature (GraphRicciCurvature)

NOTE: this is the frozen version that produced everything in plots/ -- kept
verbatim for reproducibility. The main repo's pipeline_draft.py has since
evolved (different Pipeline signature); do not mix the two.
"""

import os

# networkit (used by GraphRicciCurvature for all-pairs-shortest-path) bundles
# its own OpenMP runtime; with torch/scipy/sklearn each bundling their own too,
# multiple OpenMP runtimes threading in one process segfaults on macOS. Forcing
# single-threaded OpenMP here sidesteps the conflict. Must be set before any
# native lib below is imported -- OpenMP reads it at library init time.
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import multiprocessing as mp

# GraphRicciCurvature hands its state to Pool workers via module globals, which
# only survive process *fork*. macOS defaults to *spawn* since py3.8, where the
# workers start blank and die on NameError -- so force fork (safe here: OpenMP
# is pinned to one thread above, so no forked-thread deadlock).
try:
    mp.set_start_method('fork')
except RuntimeError:
    pass  # already set by the embedding process

from transformer_lens import HookedTransformer
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from ripser import ripser
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import persim
from sklearn.neighbors import radius_neighbors_graph
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import skdim

from null_cloud import Manifold
from topology_metric import TopologyMetric


class Pipeline():
    def __init__(self, pos_prompts):
        # bfloat16: fp32 gemma-2-2b is ~10.5GB with a ~2x peak while transformer_lens
        # converts the HF weights -- too much for a 16GB machine. bf16 halves it.
        #
        # Weights come from unsloth's ungated mirror of google/gemma-2-2b (the
        # official repo is licence-gated and needs a logged-in HF account);
        # transformer_lens still uses its own 'gemma-2-2b' config + weight
        # conversion, we just hand it the externally loaded HF model.
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf_model = AutoModelForCausalLM.from_pretrained('unsloth/gemma-2-2b',
                                                        torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained('unsloth/gemma-2-2b')
        self.model = HookedTransformer.from_pretrained('gemma-2-2b', hf_model=hf_model,
                                                       tokenizer=tokenizer,
                                                       dtype=torch.bfloat16)
        del hf_model  # free the HF copy once transformer_lens has converted it
        self.pos_prompts = pos_prompts

    def collect_activations(self): #returns a dict filled with matrices of final-token activations for each layer, for each prompt
        '''Final-token resid_post activations for every layer: {layer_name: [n, d]}.'''
        activations_dict = {f'layer_{l + 1}':[] for l in range(self.model.cfg.n_layers)} #Dict: {"layer_{layer number}": [final activations for each prompt]}

        for prompt in self.pos_prompts: #loops through each prompt
            with torch.no_grad():
                _, cache = self.model.run_with_cache(prompt)
                for l in range(self.model.cfg.n_layers):
                    activations_dict[f'layer_{l+1}'].append(cache['resid_post', l][0, -1, :]) #adds the final token's activation for each layer for current prompt to the dict

        for layer, _ in activations_dict.items():
            # .float(): numpy has no bfloat16, so cast up before .numpy()
            activations_dict[layer] = torch.stack(activations_dict[layer]).detach().cpu().float().numpy() #compacts each layer's activations for all prompts into a single tensor and converts to numpy

        return activations_dict 


    def reduce_pca(self, final_activations, var_threshold=0.95): #returns list of pca vectors for 95% variance
        #Analysis reduction: keep enough components to capture the concept.
        X = final_activations.detach().cpu().numpy() if isinstance(final_activations, torch.Tensor) else final_activations #ensures final_activations is a numpy array
        pca = PCA(n_components=min(X.shape)) #cap on how many components to keep (min of number of points and number of dimensions) (if prompts > 2304 then PCA will only keep 2304 components)
        full = pca.fit_transform(X) #all prompt activations projected into PCA space [n_prompts, n_components] by decreasing variance for each component
        m = int(np.searchsorted(np.cumsum(pca.explained_variance_ratio_), var_threshold)) + 1 #number of components to keep (m) such that the cumulative variance explained by the first m components is at least var_threshold (default 0.95)
        print(f'PCA: keeping m={m} components ({var_threshold*100:.0f}% variance)')
        return full[:, :m] #first m pcas that capture at least var_threshold variance

    def plot_pca_2D(self, final_activations): #plots 2D PCA of final activations
        #Visualization only: 2-D projection
        X = final_activations.detach().cpu().numpy() if isinstance(final_activations, torch.Tensor) else final_activations
        projected = PCA(n_components=2).fit_transform(X)

        plt.figure()
        plt.scatter(projected[:, 0], projected[:, 1])
        plt.xlabel('PC1')
        plt.ylabel('PC2')
        plt.title('PCA of layer activations')
        plt.savefig('pca_projection.png', dpi=150, bbox_inches='tight')
        print('Saved PCA plot to pca_projection.png')
        plt.show()

    def plot_pca_3D(self, final_activations): #plots 3D PCA of final activations
        #Visualization only: 3-D projection
        X = final_activations.detach().cpu().numpy() if isinstance(final_activations, torch.Tensor) else final_activations
        projected = PCA(n_components=3).fit_transform(X)
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2])
        ax.set_xlabel('PC1')
        ax.set_ylabel('PC2')
        ax.set_zlabel('PC3')
        ax.set_title('3D PCA of layer activations')
        plt.savefig('pca_projection.png', dpi=150, bbox_inches='tight')
        print('Saved PCA plot to pca_projection.png')
        plt.show()

        return projected

    def get_intrinsic_dim(self, contrastive_diff): #returns ID using TwoNN
        X = contrastive_diff.numpy() if isinstance(contrastive_diff, torch.Tensor) else contrastive_diff
        d = skdim.id.TwoNN().fit(X).dimension_
        print(f'Intrinsic dimension (TwoNN): {d:.2f}')
        return d 

    def create_persistence_diagram(self, projected):   #persistent homology from (eps = 1 to inf)
        persist_diagram = ripser(projected, maxdim = 1, distance_matrix=False, do_cocycles = False, n_perm = None )
        return persist_diagram #persistence diagram of the projected points, showing the birth and death of topological features as epsilon increases

    def create_epsilon_graph(self,projected, eps): #nxgraph with the edges for a specified epsilon value
        graph = radius_neighbors_graph(projected,radius = eps,mode = 'distance', metric='euclidean') #nxn matrix of weights that connect edges
        '''
        mode = 'distance' ensures that the graph is not binary (when all distances are 1.0)
        '''

        graph = nx.Graph(graph)

        return graph 

    def compute_ollivier_ricci(self, graph):  #returns {'graph': networkx graph with curvature values, 'mean_curvature': mean curvature(double), 'raw_values': list of curvature values for each edge in the graph}
        orc = OllivierRicci(graph, alpha = 0.5, proc = 1, verbose = 'ERROR')
        orc_curv = orc.compute_ricci_curvature()

        raw_values = []
        for edge in orc_curv.edges(data='ricciCurvature'):
            raw_values.append(edge[-1]) #structure for each edge between points (u,v) is (u,v,curvature_value)

        mean_curv = np.mean(raw_values)

        summ_dict = {'graph': orc_curv, 'mean_curvature': mean_curv, 'raw_values':raw_values}

        return summ_dict
