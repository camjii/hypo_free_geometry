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
        self.model = HookedTransformer.from_pretrained('gpt-2')
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
            mean = np.abs(np.mean(diff))

            if mean > max_diff_mean:
                max_diff_mean, opt_layer, contrastive_diff = mean, layer, diff

        print(f'The layer with the greatest mean vector is layer {opt_layer} with a mean of {max_diff_mean}')
        contrastive_diff.detach().cpu().numpy()

        return opt_layer, contrastive_diff

              
    
    '''
    Replace the create_distance_matrix function with running PCA on contrastive_diff

    Figure out how a persistence diagram can be created from PCA
    '''
    
    
    def plot_pca(self, contrastive_diff):
        pca = PCA(n_components=3)
        projected = pca.fit_transform(contrastive_diff)

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection='3d')
        sc = ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2],
                         c=projected[:, 2], cmap='viridis',
                         s=50, alpha=0.85, edgecolors='white', linewidths=0.5)
        fig.colorbar(sc, ax=ax, label='PC3', shrink=0.6, pad=0.1)

        var_ratio = pca.explained_variance_ratio_
        ax.set_xlabel(f'PC1 ({var_ratio[0]*100:.1f}%)')
        ax.set_ylabel(f'PC2 ({var_ratio[1]*100:.1f}%)')
        ax.set_zlabel(f'PC3 ({var_ratio[2]*100:.1f}%)')
        ax.set_title('PCA of contrastive activation differences')
        fig.tight_layout()
        plt.show()

        return projected
    
    def get_intrinsic_dim(self, contrastive_diff):
        X = contrastive_diff.numpy() if isinstance(contrastive_diff, torch.Tensor) else contrastive_diff
        d = skdim.id.TwoNN().fit(X).dimension_
        print(f'Intrinsic dimension (TwoNN): {d:.2f}')
        return d #returns ID using TwoNN
    
    def create_persistence_diagram(self, projected):   #persistent homology from (eps = 1 to inf)
        persist_diagram = ripser(projected, maxdim = 1, distance_matrix=True, do_cocycles = False, n_perm = None )
        return persist_diagram
    
    def create_epsilon_graph(self,projected, eps):
        graph = radius_neighbors_graph(projected,radius = eps,mode = 'distance', metric='precomputed') #nxn matrix of weights that connect edges
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
    

    
