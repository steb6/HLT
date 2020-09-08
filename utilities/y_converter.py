import numpy as np

y_label_to_category_map = {'mixed': 0,
                           'negative': 1,
                           'positive': 2}

y_category_to_onehot = {0: [1, 0, 0],
                        1: [0, 1, 0],
                        2: [0, 0, 1]}


def onehot_to_category(results):
    pruned = []
    for result in results:
        index = np.where(result == max(result))[0][0]
        pruned.append(index)
    return pruned
