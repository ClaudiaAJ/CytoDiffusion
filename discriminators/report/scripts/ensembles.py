"""Does adding a hyperbolic-head model to an ensemble help more than adding another Euclidean (linear-head) model?
Probability averaging of fine-tuned ViT-B/16 models (same data, same training loop; heads differ)."""
import numpy as np, itertools, glob, os
from sklearn.metrics import balanced_accuracy_score
O = '/store/CIA/ca662/CytoDiffusion_runs/hyp_accuracy'
def sm(l):
    l = l - l.max(1, keepdims=True); e = np.exp(l); return e / e.sum(1, keepdims=True)
M = {}
for f in sorted(glob.glob(f'{O}/ft_*.npz')):
    d = np.load(f); n = os.path.basename(f)[3:-4]
    M[n] = dict(v=sm(d['val_logits']), t=sm(d['test_logits']), yv=d['val_y'], yt=d['test_y'])
yt = next(iter(M.values()))['yt']; yv = next(iter(M.values()))['yv']
def score(names, split='t'):
    p = np.mean([M[n][split] for n in names], 0).argmax(1); y = yt if split == 't' else yv
    return (p == y).mean() * 100, balanced_accuracy_score(y, p) * 100
print('single models (test acc / balanced acc | val acc)')
for n in M:
    a, b = score([n]); print(f'  {n:22s} {a:5.1f} / {b:5.1f} | {score([n], "v")[0]:5.1f}')
L = [n for n in M if n.startswith('linear')]; P = [n for n in M if n.startswith('poincare_clip')]
def group(label, combos):
    if not combos: return
    s = np.array([score(c) for c in combos]); v = np.array([score(c, 'v') for c in combos])
    print(f'  {label:44s} n={len(combos):2d}  test acc {s[:,0].mean():5.1f} +- {s[:,0].std():.1f}  bal {s[:,1].mean():5.1f} +- {s[:,1].std():.1f}  | val acc {v[:,0].mean():5.1f}')
print('\nfine-tuned models from the same loop (3 seeds each): linear head vs hard-clip Poincare head')
group('1 linear', [[a] for a in L]); group('1 poincare', [[a] for a in P])
group('2 linear', list(itertools.combinations(L, 2))); group('2 poincare', list(itertools.combinations(P, 2)))
group('1 linear + 1 poincare (different seeds)', [[a, b] for a in L for b in P if a[-1] != b[-1]])
group('1 linear + 1 poincare (same seed)', [[a, b] for a in L for b in P if a[-1] == b[-1]])
group('3 linear', [L]); group('3 poincare', [P])
group('2 linear + 1 poincare', [list(c) + [p] for c in itertools.combinations(L, 2) for p in P])
group('1 linear + 2 poincare', [[l] + list(c) for l in L for c in itertools.combinations(P, 2)])
group('3 linear + 3 poincare', [L + P])
print('\nwith the stock ViT (single seed, 88.6%)')
group('stock + 1 linear', [['stock', a] for a in L]); group('stock + 1 poincare', [['stock', a] for a in P])
if 'poincare_runnorm_s0' in M: group('stock + poincare_runnorm_s0', [['stock', 'poincare_runnorm_s0']])
group('stock + 3 linear', [['stock'] + L]); group('stock + 3 poincare', [['stock'] + P]); group('all models', [list(M)])
