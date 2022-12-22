from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt

# algorithms
algos = ['HEFTAddEnd', 'HEFTAddBetween', 'Topological',
         'Genetic[generations=50,size_selection=None,mutate_order=None,mutate_resources=None]']

# algo -> blocks received
global_algo_frequencies = defaultdict(int)
# algo, block type -> blocks received
global_algo_block_type_frequencies = defaultdict(lambda: defaultdict(int))
# algo, block graph type -> blocks received
global_algo_bg_type_frequencies = defaultdict(lambda: defaultdict(int))

# algo -> downtime
global_algo_downtimes = defaultdict(int)
# algo, block type -> downtime
global_algo_block_type_downtimes = defaultdict(lambda: defaultdict(int))
# algo, block graph type -> downtime
global_algo_bg_type_downtimes = defaultdict(lambda: defaultdict(int))

run_mode_index = 0

algo_labels = ['HEFTAddEnd', 'HEFTAddBetween', 'Topological',
               'Genetic[\ngenerations=50,size_selection=None,\nmutate_order=None,mutate_resources=None]']

# parsing raw data
for run_index in range(5):
    with open(f'algorithms_comparison_block_size_{run_mode_index}_{run_index}.txt', 'r') as f:
        mode = None
        finished = True
        bg_info_read = False
        bg_info = []
        downtimes = []
        i = 0  # index of algorithm
        iter = 0  # index of iteration

        for line in f.readlines():
            line = line.strip()
            if finished:
                # skip lines between blocks
                if len(line) == 0:
                    continue
                finished = False
                bg_info_read = False

                iter += 1
                if iter == 1:
                    mode = line
                else:
                    bg_info = line.split(' ')
                    bg_info_read = True
                    if iter == 4:
                        iter = 0

                i = 0
                continue
            if not bg_info_read:
                # read bg type and block types
                bg_info = line.split(' ')
                bg_info_read = True
                continue
            if i == 10 or (i == 7 and bg_info[0] == 'Queues'):
                finished = True
                downtimes = [int(downtime) for downtime in line.split(' ')]
                for algo_ind, algo in enumerate(algos):
                    global_algo_downtimes[algo] += downtimes[algo_ind]
                    global_algo_bg_type_downtimes[bg_info[0]][algo] += downtimes[algo_ind]
                i += 1
                continue

            # in this place the line = algo name
            algo = line.replace('Scheduler', '')
            # grab statistics
            global_algo_frequencies[algo] += 1
            global_algo_bg_type_frequencies[bg_info[0]][algo] += 1
            if len(bg_info) != 1:
                global_algo_block_type_frequencies[algo][bg_info[i + 1]] += 1
            i += 1


# general comparison
def compare_algos_general(title: str, compare_dict: Dict):
    centers = range(len(algo_labels))
    _, ax = plt.subplots(figsize=(10, 7))
    ax.bar(centers, compare_dict.values(), tick_label=algo_labels)
    plt.suptitle(title, fontsize=24)
    plt.show()


compare_algos_general('Algorithms block receive count - average', global_algo_frequencies)
compare_algos_general('Algorithms downtimes - average', global_algo_downtimes)


# block graph type comparison
def compare_algos_bg_type(title: str, compare_dict: Dict):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=32)

    freq = list(compare_dict.items())[0]
    ax1 = fig.add_subplot(221)
    ax1.set(title=freq[0], xticks=[], yticks=[])
    ax1.bar(range(len(algo_labels)), [freq[1][algo] for algo in algos], tick_label=algo_labels)
    freq = list(compare_dict.items())[1]
    ax2 = fig.add_subplot(222)
    ax2.set(title=freq[0], xticks=[], yticks=[])
    ax2.bar(range(len(algo_labels)), [freq[1][algo] for algo in algos], tick_label=algo_labels)
    freq = list(compare_dict.items())[2]
    ax3 = fig.add_subplot(223)
    ax3.set(title=freq[0], xticks=[], yticks=[])
    ax3.bar(range(len(algo_labels)), [freq[1][algo] for algo in algos], tick_label=algo_labels)
    freq = list(compare_dict.items())[3]
    ax4 = fig.add_subplot(224)
    ax4.set(title=freq[0], xticks=[], yticks=[])
    ax4.bar(range(len(algo_labels)), [freq[1][algo] for algo in algos], tick_label=algo_labels)

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()


compare_algos_bg_type('Received blocks - algorithms with graph types comparison', global_algo_bg_type_frequencies)
compare_algos_bg_type('Downtimes - algorithms with graph types comparison', global_algo_bg_type_downtimes)


# block type comparison
def compare_algos_block_type(title: str, compare_dict: Dict):
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(title, fontsize=32)
    freq = list(compare_dict.items())[0]
    ax1 = fig.add_subplot(221)
    ax1.set(title=algo_labels[0], xticks=[], yticks=[])
    ax1.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
    freq = list(compare_dict.items())[1]
    ax2 = fig.add_subplot(222)
    ax2.set(title=algo_labels[1], xticks=[], yticks=[])
    ax2.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
    freq = list(compare_dict.items())[2]
    ax3 = fig.add_subplot(223)
    ax3.set(title=algo_labels[2], xticks=[], yticks=[])
    ax3.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
    freq = list(compare_dict.items())[3]
    ax4 = fig.add_subplot(224)
    ax4.set(title=algo_labels[3], xticks=[], yticks=[])
    ax4.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))

    plt.subplots_adjust(left=0.1,
                        bottom=0.1,
                        right=0.9,
                        top=0.9,
                        wspace=0.4,
                        hspace=0.4)
    plt.show()


compare_algos_block_type('Received blocks - algorithms with block types comparison', global_algo_block_type_frequencies)
