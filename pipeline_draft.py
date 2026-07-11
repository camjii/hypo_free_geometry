"""
This pipeline does the following:

1. Extracts model activations for a given feature
2. Constructs a distance matrix
3. Build an epsilon neighborhood graph, which is essentially takes the distance between each point in the distance matrix, and draws a connection between them if d(Xi, Xj) <= ε (Sk_learn, radius_neighbors_graph)
4. Runs the graph through PH (ripser.py, giotto-tda, VietorisRipsPersistence), builds on it with higher simplicial complexes, and gets a persistence diagram, showing what features upheld through all tests, and where structures dissipated
5. Get the curvature signature of the ε-graph via ollivier ricci curvature (GraphRicciCurvature)
"""

import transformer_lens
from transformer_lens import HookedTransformer
import networkx as nx
from GraphRicciCurvature.OllivierRicci import OllivierRicci
from ripser import ripser
import numpy as np
import torch
from sklearn.metrics.pairwise import cosine_distances
import persim
from sklearn.neighbors import radius_neighbors_graph
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


        return opt_layer, contrastive_diff

              
    def create_distance_matrix(self, contrastive_diff):
        dist_matrix = cosine_distances(contrastive_diff) #gets distance matrix in one vectorized call 
        return dist_matrix

    
    def create_persistence_diagram(self, dist_matrix):   
        persist_diagram = ripser(dist_matrix, maxdim = 1, distance_matrix=True, do_cocycles = False, n_perm = None )
        return persist_diagram
    
    def create_epsilon_graph(self,dist_matrix):
        graph = radius_neighbors_graph(dist_matrix,)

    #WIP

       
