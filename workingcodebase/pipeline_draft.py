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

    def collect_activations(self):
        '''Final-token resid_post activations for every layer: {layer_name: [n, d]}.'''
        activations_dict = {f'layer_{l + 1}':[] for l in range(self.model.cfg.n_layers)}

        for prompt in self.pos_prompts:
            with torch.no_grad():
                _, cache = self.model.run_with_cache(prompt)
                for l in range(self.model.cfg.n_layers):
                    activations_dict[f'layer_{l+1}'].append(cache['resid_post', l][0, -1, :])

        for layer, _ in activations_dict.items():
            # .float(): numpy has no bfloat16, so cast up before .numpy()
            activations_dict[layer] = torch.stack(activations_dict[layer]).detach().cpu().float().numpy()

        return activations_dict

    def plot_layer_grid(self, activations_dict, point_labels=None, path='layer_grid.png'):
        '''
        One PCA scatter per layer, for handpicking the layer with the cleanest
        representation. Each panel is annotated with two circle diagnostics:

          CV    std(radius)/mean(radius) of the 2D PCA points -- 0 for a perfect
                circle, larger as the shape degrades toward a blob or a line.
          order 'o' if walking around the centroid by angle visits the points in
                their true cyclic order (any rotation, either direction).
        '''
        n_layers = len(activations_dict)
        cols = 6
        rows = int(np.ceil(n_layers / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(3.2 * cols, 3.2 * rows))

        scores = {}
        for ax, (layer, act) in zip(np.ravel(axes), activations_dict.items()):
            X = PCA(n_components=2).fit_transform(act)

            r = np.linalg.norm(X, axis=1)          # PCA output is centered
            cv = float(r.std() / r.mean())

            angles = np.arctan2(X[:, 1], X[:, 0])
            walk = list(np.argsort(angles))         # visit order around the centroid
            n_pts = len(walk)
            cyclic_ok = any(walk == [(s + k * d) % n_pts for k in range(n_pts)]
                            for s in range(n_pts) for d in (1, -1))
            scores[layer] = {'radial_cv': cv, 'cyclic_order': cyclic_ok}

            ax.scatter(X[:, 0], X[:, 1], s=18)
            if point_labels is not None:
                for (x, y), lab in zip(X, point_labels):
                    ax.annotate(lab, (x, y), fontsize=7, alpha=0.8,
                                textcoords='offset points', xytext=(3, 3))
            ax.set_title(f'{layer}  CV={cv:.2f}' + ('  o' if cyclic_ok else ''),
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])

        for ax in np.ravel(axes)[n_layers:]:
            ax.axis('off')

        fig.suptitle('Per-layer PCA of final-token activations '
                     '(CV: 0 = perfect circle; o = correct cyclic order)', y=1.0)
        fig.tight_layout()
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'Saved layer grid to {path}')

        # ranking hint: order-correct layers first, then roundest
        ranked = sorted(scores, key=lambda l: (not scores[l]['cyclic_order'],
                                               scores[l]['radial_cv']))
        print('Most circular first:', ranked[:5])
        return scores

    def select_layer_by_topology(self, kind='noise', n_null=3, max_dim=1,
                                 activations_dict=None):
        '''
        Layer selection by shape, not magnitude: score each layer by how far its
        activation cloud sits from its own null. Pass activations_dict to reuse
        already-collected activations instead of re-running the model.
        '''
        if activations_dict is None:
            activations_dict = self.collect_activations()

        best_layer, best_score, scores = None, -np.inf, {}
        for layer, act in activations_dict.items():
            manifold = Manifold(self, act, np.zeros_like(act), cloud=act, label=layer)
            tm = TopologyMetric(manifold, kind=kind, n_null=n_null, max_dim=max_dim)
            scores[layer] = tm.metric

            # topology is [H0, H1] bottleneck vs null; H1 (loop structure) is the
            # signal layer selection previously scored on when max_dim defaulted to 1.
            h1_score = tm.topology[1]
            if h1_score > best_score:
                best_score, best_layer = h1_score, layer

        print(f'Layer with the strongest topology-vs-null signal is {best_layer} (topology={best_score:.4f})')

        return best_layer, activations_dict[best_layer], scores

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
        plt.savefig('pca_projection.png', dpi=150, bbox_inches='tight')
        print('Saved PCA plot to pca_projection.png')
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
