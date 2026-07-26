import numpy as np
from pipeline_draft import Pipeline

def format_prompt(day):
    return f'The day of the week is {day}'

day_list = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

prompts_list = []

for day in day_list:
    prompts_list.append(format_prompt(day))

if __name__ == '__main__':
    pipeline = Pipeline(prompts_list)

    # every layer's PCA in one figure, for handpicking the cleanest circle
    activations_dict = pipeline.collect_activations()
    circle_scores = pipeline.plot_layer_grid(activations_dict,
                                             point_labels=[d[:3] for d in day_list])

    # topology-vs-null selection still available for comparison:
    # opt_layer, opt_activations, scores = pipeline.select_layer_by_topology()
    # print(scores)
    # pipeline.plot_pca(opt_pos_activations=opt_activations)
