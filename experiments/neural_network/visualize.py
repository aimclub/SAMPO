import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import transforms

a = [98.2785, 98.6533, 241.6574, 328.5511, 402.8818, 145.0562, 305.0154, 170.0357, 119.5020]
labels = []
name = 'Graph parallelism degree '
for i in range(len(a)):
    labels.append(name + str(i))

weights = np.asarray([230.9019, 342.5922, 341.3734, 476.5591, 98.2785, 98.6533, 241.6574, 328.5511, 402.8818, 145.0562, 305.0154, 170.0357, 119.5020])
labels = ['Vertex amount', 'Average work per activity', 'Relative max children', 'Average resource usage', 'Graph parallelism degree 0',
 'Graph parallelism degree 1',
 'Graph parallelism degree 2',
 'Graph parallelism degree 3',
 'Graph parallelism degree 4',
 'Graph parallelism degree 5',
 'Graph parallelism degree 6',
 'Graph parallelism degree 7',
 'Graph parallelism degree 8']

dt = {}
for i in range(len(labels)):
 dt[labels[i]] = weights[i]

dt = dict(reversed(sorted(dt.items(), key=lambda item: item[1])))

df = pd.DataFrame(data=dt, index=[0])
tr = transforms.Affine2D().rotate_deg(90)
weights2 = weights.reshape(weights.shape[0], 1)
# fig, ax = plt.subplots()
sns.heatmap(df.T, square=True, xticklabels=['importance'])
# ax.invert_yaxis()
# plt.show()
plt.savefig('foo.png')