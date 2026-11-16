"""Pooling of ViT tokens in Euclidean vs hyperbolic space, then a prototype-distance classifier.
Per token: LayerNorm -> Linear(768, d) -> running-norm scaling (mean token norm = 1). Pool. Running-norm scaling of the pooled vector. Prototypes.
variants:
  cls_euclid / cls_poincare   class token only (reference)
  mean_euclid                 plain mean of all token vectors
  normw_euclid                Euclidean mean with the Einstein-midpoint weights (Lorentz factors of the tokens) -> isolates the weighting
  midpoint_poincare           Einstein midpoint of the tokens on the ball (tokens far from the origin weigh more), hyperbolic prototypes
  attn_euclid                 learned attention pooling (softmax of a learned score), Euclidean mean
  attn_poincare               the same learned attention weights inside the Einstein midpoint (hyperbolic attention, Gulcehre et al. 2019)
"""
import sys, argparse, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from sklearn.metrics import balanced_accuracy_score
sys.path.insert(0, '/mhome/damtp/t/ca662/CytoDiffusion/discriminators')
import importlib.util
spec = importlib.util.spec_from_file_location("hyp", "/mhome/damtp/t/ca662/CytoDiffusion/discriminators/train_and_test_hyperbolic.py"); hyp = importlib.util.module_from_spec(spec); spec.loader.exec_module(hyp)

class RunNorm(nn.Module):
    def __init__(self, target=1.0):
        super().__init__(); self.target = target
        self.register_buffer("m", torch.ones(())); self.register_buffer("init", torch.zeros(()))
    def forward(self, v):
        if self.training:
            with torch.no_grad():
                n = v.norm(dim=-1).mean()
                if self.init.item() == 0: self.m.copy_(n); self.init.fill_(1)
                else: self.m.mul_(0.9).add_(0.1 * n)
        return v * (self.target / self.m.clamp_min(1e-6))

class TokenHead(nn.Module):
    def __init__(self, variant, n_classes, dim=64, c=1.0, tau=0.1, clip_r=4.0):
        super().__init__()
        self.variant, self.c, self.clip_r = variant, c, clip_r
        self.hyper = variant.endswith("poincare")
        self.ln = nn.LayerNorm(768, elementwise_affine=False); self.proj = nn.Linear(768, dim)
        with torch.no_grad(): self.proj.weight.mul_(0.3); self.proj.bias.zero_()
        self.tok_norm, self.pool_norm = RunNorm(), RunNorm()
        if variant.startswith("attn"): self.score = nn.Linear(dim, 1)
        p = torch.randn(n_classes, dim); self.proto = nn.Parameter(p / p.norm(dim=-1, keepdim=True))
        self.log_inv_tau = nn.Parameter(torch.tensor(float(np.log(1 / tau))))
    def _clip(self, v):
        return v * torch.clamp(self.clip_r / v.norm(dim=-1, keepdim=True).clamp_min(1e-9), max=1.0)
    def _log0(self, z):
        sc = self.c ** 0.5; n = z.norm(dim=-1, keepdim=True).clamp_min(1e-9); n = torch.minimum(n, (1 - 1e-5) / sc * torch.ones_like(n))
        return torch.atanh(sc * n) * z / (sc * n)
    def embed(self, tok):
        c = self.c
        if self.variant.startswith("cls"): tok = tok[:, :1]
        v = self._clip(self.tok_norm(self.proj(self.ln(tok.float()))))            # (B,T,d) tangent vectors
        w = torch.softmax(self.score(v).squeeze(-1), dim=1) if self.variant.startswith("attn") else None
        if self.variant in ("mean_euclid", "cls_euclid", "attn_euclid"):
            pooled = v.mean(1) if w is None else (w[..., None] * v).sum(1)
        else:
            x = hyp.expmap0(v, c)
            k = 2 * x / (1 + c * x.pow(2).sum(-1, keepdim=True))                    # Klein coordinates
            g = 1 / (1 - c * k.pow(2).sum(-1, keepdim=True)).clamp_min(1e-6).sqrt()  # Lorentz factors
            if w is not None: g = g * w[..., None]
            if self.variant == "normw_euclid":
                pooled = (g * v).sum(1) / g.sum(1)
            else:
                m = (g * k).sum(1) / g.sum(1)
                mp = m / (1 + (1 - c * m.pow(2).sum(-1, keepdim=True)).clamp_min(1e-6).sqrt())  # back to Poincare
                pooled = self._log0(mp)
        pooled = self._clip(self.pool_norm(pooled))
        return hyp.expmap0(pooled, c) if self.hyper else pooled
    def forward(self, tok):
        z = self.embed(tok)
        if self.hyper: d = hyp.poincare_dist(z, hyp.expmap0(self.proto, self.c), self.c)
        else: d = torch.cdist(z, self.proto)
        return -d * self.log_inv_tau.exp()

@torch.no_grad()
def predict(head, X, bs=256):
    head.eval(); return torch.cat([head(X[i:i + bs]).argmax(1) for i in range(0, len(X), bs)])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tokens"); p.add_argument("--variants", nargs="+"); p.add_argument("--dim", type=int, default=64)
    p.add_argument("--c", type=float, default=1.0); p.add_argument("--steps", type=int, default=900); p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4]); p.add_argument("--out", default=None)
    a = p.parse_args(); dev = torch.device("cuda:0")
    D = {}
    for s in ("train", "val", "test"):
        D[s] = (torch.from_numpy(np.load(f"{a.tokens}/{s}_tok.npy")).to(dev), torch.from_numpy(np.load(f"{a.tokens}/{s}_y.npy")).to(dev))
    Xtr, ytr = D["train"]; preds = {}
    for var in a.variants:
        res = []
        for seed in a.seeds:
            torch.manual_seed(seed); head = TokenHead(var, 10, a.dim, a.c).to(dev)
            opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=1e-2); best = (-1, 0, 0, None)
            for step in range(1, a.steps + 1):
                head.train(); idx = torch.randperm(len(Xtr), device=dev)[:32]
                loss = F.cross_entropy(head(Xtr[idx]), ytr[idx])
                if not torch.isfinite(loss): print(f"  [{var}] non-finite loss at step {step}"); break
                opt.zero_grad(); loss.backward(); opt.step()
                if step % 50 == 0:
                    pv = predict(head, D["val"][0]); va = (pv == D["val"][1]).float().mean().item() * 100
                    if va >= best[0]:
                        pt = predict(head, D["test"][0]); yt = D["test"][1]
                        best = (va, (pt == yt).float().mean().item() * 100, balanced_accuracy_score(yt.cpu().numpy(), pt.cpu().numpy()) * 100, pt.cpu().numpy())
            res.append(best[:3]); preds[f"{var}_s{seed}"] = best[3]
        r = np.array(res)
        print(f"  {var:18s} test@bestval {r[:,1].mean():5.1f}+-{r[:,1].std():.1f}  balanced {r[:,2].mean():5.1f}+-{r[:,2].std():.1f}  | valacc@best {r[:,0].mean():5.2f}", flush=True)
    if a.out: np.savez(a.out, y=D["test"][1].cpu().numpy(), **preds)

if __name__ == "__main__":
    main()
