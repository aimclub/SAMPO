from sampo.generator import SimpleSynthetic

from tqdm import tqdm

ss = SimpleSynthetic(rand=231)

for size in range(100, 500 + 1, 100):
    for i in tqdm(range(100)):
        wg = ss.work_graph(bottom_border=size - 5,
                           top_border=size)
        while not (size - 5 <= wg.vertex_count <= size):
            wg = ss.work_graph(bottom_border=size - 20,
                               top_border=size)
        wg.dump('wgs', f'{size}_{i}')
