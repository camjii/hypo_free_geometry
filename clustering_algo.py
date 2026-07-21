'''
Run the pipeline (get point cloud) -> create manifold (Manifold) -> run TopologyMetric -> get point -> accumulate points in dictionary {concept:point} -> plot on graph
'''

from pipeline_draft import Pipeline
from compare import ManifoldComparator
from null_cloud import Manifold
from topology_metric import TopologyMetric

#train an autoencoder?  






class ClusteringAlgo():

    def __init__(self, concepts):
        '''
        concepts: dict[str, list[str]] mapping concept name -> prompts defining it
        '''
        self.concepts = concepts
        self.points = {}   # concept -> point (TopologyMetric.metric)

    def get_point(self, name, prompts):
        '''
        pipeline -> point cloud -> Manifold -> TopologyMetric -> point
        (select_layer_by_topology already builds Manifold + TopologyMetric per
        layer internally -- check scores[layer] before recomputing either)
        '''
        pipeline = Pipeline(prompts)
        opt_layer, opt_activations, scores = pipeline.select_layer_by_topology()
        manifold = Manifold(pipeline, opt_activations)
        null = Manifold.null()

        metric = TopologyMetric(manifold)
        self.points['name'] = metric
        return metric.getMetric()
    

    def clustering_metric():


        pass

if __name__ == '__main__':
    pass
