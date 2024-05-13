import json
from itertools import chain
from random import Random

import matplotlib.pyplot as plt

with open('hybrid_results.json', 'r') as f:
    results = json.load(f)

results = {int(graph_size): [100 * (1 - res) for res in chain(*result)]
           for graph_size, result in results.items()}
graph_sizes = results.keys()

rand = Random()

plt.title('Прирост качества планов\nот применения гибридизации', fontsize=16)
plt.xlabel('Размер графа')
plt.ylabel('% прироста качества относительно базового алгоритма')
plt.boxplot(results.values(), labels=graph_sizes)
plt.show()
