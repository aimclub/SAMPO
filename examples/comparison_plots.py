from collections import defaultdict

import matplotlib.pyplot as plt

# algo -> blocks received
global_algo_frequencies = defaultdict(int)
# algo, block type -> blocks received
global_algo_block_type_frequencies = defaultdict(lambda: defaultdict(int))
global_algo_bg_type_frequencies = defaultdict(lambda: defaultdict(int))

with open('comparison.txt', 'r') as f:
    mode = None
    finished = True
    bg_info_read = False
    bg_info = []
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

            if iter == 0:
                mode = line
                iter += 1
            else:
                bg_info = line.split(' ')
                bg_info_read = True
                if iter == 3:
                    iter = 0

            i = 0
            continue
        if not bg_info_read:
            # read bg type and block types
            bg_info = line.split(' ')
            bg_info_read = True
            continue
        if len(line) == 0:
            finished = True
            continue

        if 'Scheduler' in line:
            # in this place the line = algo name
            algo = line.replace('Scheduler', '')
            # grab statistics
            global_algo_frequencies[algo] += 1
            global_algo_bg_type_frequencies[bg_info[0]][algo] += 1
            if len(bg_info) != 1:
                global_algo_block_type_frequencies[algo][bg_info[i + 1]] += 1
            i += 1

# general comparison
algo_labels = list(global_algo_frequencies.keys())
algo_freq = list(global_algo_frequencies.values())

centers = range(len(algo_labels))
plt.suptitle('Algorithms block receive count', fontsize=24)
plt.bar(centers, algo_freq, tick_label=algo_labels)

plt.show()

# block graph type comparison
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Algorithms with graph types comparison', fontsize=32)
freq = list(global_algo_bg_type_frequencies.items())[0]
ax1 = fig.add_subplot(221)
ax1.set(title=freq[0], xticks=[], yticks=[])
ax1.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
freq = list(global_algo_bg_type_frequencies.items())[1]
ax2 = fig.add_subplot(222)
ax2.set(title=freq[0], xticks=[], yticks=[])
ax2.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
freq = list(global_algo_bg_type_frequencies.items())[2]
ax3 = fig.add_subplot(223)
ax3.set(title=freq[0], xticks=[], yticks=[])
ax3.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
freq = list(global_algo_bg_type_frequencies.items())[3]
ax4 = fig.add_subplot(224)
ax4.set(title=freq[0], xticks=[], yticks=[])
ax4.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))

plt.show()

# block type comparison
fig = plt.figure(figsize=(14, 10))
fig.suptitle('Algorithms with block types comparison', fontsize=32)
freq = list(global_algo_block_type_frequencies.items())[0]
ax1 = fig.add_subplot(221)
ax1.set(title=freq[0], xticks=[], yticks=[])
ax1.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
freq = list(global_algo_block_type_frequencies.items())[1]
ax2 = fig.add_subplot(222)
ax2.set(title=freq[0], xticks=[], yticks=[])
ax2.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
freq = list(global_algo_block_type_frequencies.items())[2]
ax3 = fig.add_subplot(223)
ax3.set(title=freq[0], xticks=[], yticks=[])
ax3.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))
freq = list(global_algo_block_type_frequencies.items())[3]
ax4 = fig.add_subplot(224)
ax4.set(title=freq[0], xticks=[], yticks=[])
ax4.bar(range(len(freq[1])), freq[1].values(), tick_label=list(freq[1].keys()))

plt.show()

