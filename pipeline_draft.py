"""
This pipeline does the following:

1. Extracts model activations for a given feature
2. Constructs a distance matrix
3. Build an epsilon neighborhood graph, which is essentially takes the distance between each point in the distance matrix, and draws a connection between them if d(Xi, Xj) <= ε (Sk_learn, radius_neighbors_graph)
4. Runs the graph through PH (ripser.py, giotto-tda, VietorisRipsPersistence), builds on it with higher simplicial complexes, and gets a persistence diagram, showing what features upheld through all tests, and where structures dissipated
5. Get the curvature signature of the ε-graph via ollivier ricci curvature (GraphRicciCurvature)
"""

import os
os.environ['OBJC_DISABLE_INITIALIZE_FORK_SAFETY'] = 'YES'
#

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


class Pipeline():
    def __init__(self,pos_prompts, neg_prompts):
        self.model = HookedTransformer.from_pretrained('gpt2')
        self.pos_prompts = pos_prompts #get from CKA github
        self.neg_prompts = neg_prompts

    
    

  
    def get_layer(self):
        activations_dict = {'pos_activations': {f'layer_{l + 1}':[] for l in range(self.model.cfg.n_layers)}, 'neg_activations': {f'layer_{l + 1}':[] for l in range(self.model.cfg.n_layers) }} #storing activations in dictionary
        act_types = ['pos', 'neg']
        
        for type in act_types:
            prompts = self.pos_prompts if type == 'pos' else self.neg_prompts
            key = f'{type}_activations'

            for prompt in prompts:
                with torch.no_grad():
                    _, cache = self.model.run_with_cache(prompt) #getting logits + cache per prompt
                for l in range(self.model.cfg.n_layers):
                    activations_dict[key][f'layer_{l+1}'].append(cache['resid_post', l][0, -1, :]) #appending the activations at the last token to each layer
            

            for layer, _ in activations_dict[key].items():
                activations_dict[key][layer] = torch.stack(activations_dict[key][layer]) #stacking them to get matrix




            #for each layer compute difference of each matrix and get the mean
        max_diff_mean, opt_layer, contrastive_diff = 0, None, None
        for layer in activations_dict['pos_activations'].keys():

            diff = activations_dict['pos_activations'][layer] - activations_dict['neg_activations'][layer]
            mean = diff.mean(dim=0).norm().item() #norm of the mean-difference vector, not a scalar mean over all dims

            if mean > max_diff_mean:
                max_diff_mean, opt_layer, contrastive_diff = mean, layer, diff

        print(f'The layer with the greatest mean vector is layer {opt_layer} with a mean of {max_diff_mean}')
        contrastive_diff = contrastive_diff.detach().cpu().numpy()
        opt_pos_activations = activations_dict['pos_activations'][opt_layer].detach().cpu().numpy()

        return opt_layer, contrastive_diff, opt_pos_activations

              
    
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

    def plot_pca(self, contrastive_diff):
        #Visualization only: 3-D projection
        X = contrastive_diff.detach().cpu().numpy() if isinstance(contrastive_diff, torch.Tensor) else contrastive_diff
        projected = PCA(n_components=3).fit_transform(X)
        return projected

    def get_intrinsic_dim(self, contrastive_diff):
        X = contrastive_diff.numpy() if isinstance(contrastive_diff, torch.Tensor) else contrastive_diff
        d = skdim.id.TwoNN().fit(X).dimension_
        print(f'Intrinsic dimension (TwoNN): {d:.2f}')
        return d #returns ID using TwoNN

    def create_persistence_diagram(self, projected):   #persistent homology from (eps = 1 to inf)
        persist_diagram = ripser(projected, maxdim = 1, distance_matrix=False, do_cocycles = False, n_perm = None )
        return persist_diagram
    
    def create_epsilon_graph(self,projected, eps):
        graph = radius_neighbors_graph(projected,radius = eps,mode = 'distance', metric='euclidean') #nxn matrix of weights that connect edges
        '''
        mode = 'distance' ensures that the graph is not binary (when all distances are 1.0)
        '''
        
        graph = nx.Graph(graph) 
        
        return graph 
   
    def compute_ollivier_ricci(self, graph):
        orc = OllivierRicci(graph, alpha = 0.5, verbose = 'INFO') 
        orc_curv = orc.compute_ricci_curvature()

        raw_values = []
        for edge in orc_curv.edges(data='ricciCurvature'):
            raw_values.append(edge[-1]) #structure for each edge between points (u,v) is (u,v,curvature_value)
        
        mean_curv = np.mean(raw_values) 

        summ_dict = {'graph': orc_curv, 'mean_curvature': mean_curv, 'raw_values':raw_values}

        return summ_dict
    
   

