import time, sys
sys.path.insert(0, '.')
from unipret.data_bridge import PerturbationReader, select_hvg, build_samples, load_local_dataset

SCIPLEX3 = 'C:/Users/wkr20/Desktop/virtual_cell_real_data/sciPlex3/SrivatsanTrapnell2020_sciplex3.h5ad'
LOCAL = 'dataset.json'

r = PerturbationReader(SCIPLEX3, backed=True)
r.load_full_sparse()   # 模拟阶段 B：整块加载（sciPlex3 实际约 7.7GB，修正估算后可通过）
local = load_local_dataset(LOCAL)

t0 = time.time()
hvg = select_hvg(r, n=2000)
print('select_hvg 用时 %.1fs  hvg维=%d' % (time.time() - t0, len(hvg)), flush=True)

t0 = time.time()
samples = build_samples(r, local, kind='compound', hvg=hvg)
print('build_samples 用时 %.1fs  化合物样本=%d' % (time.time() - t0, len(samples)), flush=True)
if samples:
    print('示例:', samples[0]['name'], 'delta维度', samples[0]['expr_delta'].shape, flush=True)
r.close()
print('SMOKE OK', flush=True)
