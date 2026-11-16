"""Checks of the Poincare-ball maths in train_and_test_hyperbolic.py against known identities.

    python test_hyperbolic_maths.py          # prints PASS/FAIL per check
    pytest test_hyperbolic_maths.py          # same checks as pytest tests

All checks run in float64 at curvatures c = 0.1, 1, 4 on random points well inside the ball.
"""
import importlib.util
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
spec = importlib.util.spec_from_file_location("hyp", os.path.join(HERE, "train_and_test_hyperbolic.py"))
hyp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(hyp)

CS = (0.1, 1.0, 4.0)
DIM, N = 8, 64


def points(c, seed=0, max_tangent=1.5):
    """Random tangent vectors (norm up to max_tangent) and their images on the ball."""
    g = torch.Generator().manual_seed(seed)
    v = torch.randn(N, DIM, generator=g, dtype=torch.float64)
    v = v / v.norm(dim=-1, keepdim=True) * torch.rand(N, 1, generator=g, dtype=torch.float64) * max_tangent
    return v, hyp.expmap0(v, c)


def head(c):
    h = hyp.PoincareHead(16, 3, dim=DIM, c=c)
    return h.double()


def close(a, b, tol):
    return float((a - b).abs().max()) <= tol


# ---------------------------------------------------------------- expmap0
def test_expmap0_stays_inside_ball():
    """Strictly inside the ball for ANY tangent norm, in float32 and float64 (boundary guard in expmap0)."""
    for c in CS:
        for dt in (torch.float32, torch.float64):
            for scale in (1.0, 8.0, 9.5, 20.0, 1e3, 1e6):
                g = torch.Generator().manual_seed(0)
                v = torch.randn(N, DIM, generator=g, dtype=dt)
                v = v / v.norm(dim=-1, keepdim=True) * (scale / c ** 0.5)
                x = hyp.expmap0(v, c)
                assert torch.isfinite(x).all(), (c, dt, scale)
                assert float((x.norm(dim=-1) * c ** 0.5).max()) < 1, (c, dt, scale)   # in the tensor's own precision, as the code computes it


def test_expmap0_guard_inactive_in_working_range():
    """For sqrt(c)|v| <= 7.5 the guard changes nothing: output is bit-identical to the unguarded formula."""
    for c in CS:
        for dt in (torch.float32, torch.float64):
            g = torch.Generator().manual_seed(1)
            v = torch.randn(N, DIM, generator=g, dtype=dt)
            v = v / v.norm(dim=-1, keepdim=True) * torch.linspace(0.01, 7.5, N, dtype=dt)[:, None] / c ** 0.5
            n = hyp._norm(v); sc = c ** 0.5
            assert torch.equal(hyp.expmap0(v, c), torch.tanh(sc * n) * v / (sc * n)), (c, dt)


def test_frac_at_cap():
    for c in CS:
        v = torch.zeros(2, DIM, dtype=torch.float64); v[0, 0] = 1 / c ** 0.5; v[1, 0] = 9 / c ** 0.5
        D = hyp.poincare_dist(torch.zeros(1, DIM, dtype=torch.float64), hyp.expmap0(v, c), c)
        assert hyp.frac_at_cap(D, c) == 0.5, c


def test_dist_saturates_at_cap():
    """Known limit: poincare_dist clamps its atanh argument at 1 - EPS, so no distance exceeds 2*atanh(1-EPS)/sqrt(c)
    (12.2 at c=1, 6.1 at c=4). Distances are exact only below that."""
    import math
    for c in CS:
        cap = hyp.max_dist(c)
        v = torch.zeros(2, DIM, dtype=torch.float64); v[0, 0] = 5 / c ** 0.5; v[1, 0] = 9 / c ** 0.5   # true distances to origin: 10/sqrt(c), 18/sqrt(c)
        d = hyp.poincare_dist(torch.zeros(1, DIM, dtype=torch.float64), hyp.expmap0(v, c), c)[0]
        assert abs(float(d[0]) - 10 / c ** 0.5) < 1e-4, c          # below the cap: exact
        assert abs(float(d[1]) - cap) < 1e-6, c                     # above the cap: saturated


def test_expmap0_norm_formula():
    """|exp0(v)| = tanh(sqrt(c)|v|) / sqrt(c)."""
    for c in CS:
        v, x = points(c)
        assert close(x.norm(dim=-1), torch.tanh(c ** 0.5 * v.norm(dim=-1)) / c ** 0.5, 1e-10), c


def test_expmap0_keeps_direction():
    for c in CS:
        v, x = points(c)
        cos = (v * x).sum(-1) / (v.norm(dim=-1) * x.norm(dim=-1))
        assert close(cos, torch.ones_like(cos), 1e-10), c


def test_log0_inverts_expmap0():
    for c in CS:
        v, x = points(c)
        assert close(head(c).log0(x), v, 1e-8), c


def test_expmap0_gradient_finite_at_zero():
    v = torch.zeros(4, DIM, dtype=torch.float64, requires_grad=True)
    hyp.expmap0(v, 1.0).sum().backward()
    assert torch.isfinite(v.grad).all()


# ---------------------------------------------------------------- mobius_add
def test_mobius_add_identity_and_inverse():
    for c in CS:
        _, x = points(c)
        zero = torch.zeros_like(x)
        assert close(hyp.mobius_add(x, zero, c), x, 1e-12), c
        assert close(hyp.mobius_add(zero, x, c), x, 1e-12), c
        assert close(hyp.mobius_add(-x, x, c), zero, 1e-10), c


def test_mobius_add_left_cancellation():
    """(-x) (+) (x (+) y) = y."""
    for c in CS:
        _, x = points(c, 0)
        _, y = points(c, 1)
        assert close(hyp.mobius_add(-x, hyp.mobius_add(x, y, c), c), y, 1e-8), c


def test_mobius_add_stays_inside_ball():
    for c in CS:
        _, x = points(c, 0)
        _, y = points(c, 1)
        assert float(hyp.mobius_add(x, y, c).norm(dim=-1).max()) < 1 / c ** 0.5, c


# ---------------------------------------------------------------- poincare_dist
def test_dist_matches_closed_form():
    """d(x,y) = (1/sqrt(c)) arcosh(1 + 2c|x-y|^2 / ((1-c|x|^2)(1-c|y|^2)))."""
    for c in CS:
        _, x = points(c, 0)
        _, y = points(c, 1)
        num = 2 * c * torch.cdist(x, y).pow(2)
        den = (1 - c * x.pow(2).sum(-1))[:, None] * (1 - c * y.pow(2).sum(-1))[None, :]
        ref = torch.acosh(1 + num / den) / c ** 0.5
        assert close(hyp.poincare_dist(x, y, c), ref, 1e-6), c


def test_dist_symmetric_and_zero_on_diagonal():
    for c in CS:
        _, x = points(c)
        D = hyp.poincare_dist(x, x, c)
        assert close(D, D.T, 1e-8), c
        assert float(torch.diag(D).abs().max()) < 1e-5, c  # not exactly 0: the norm is clamped for a finite gradient


def test_dist_to_origin_is_twice_tangent_norm():
    """d(0, exp0(v)) = 2|v| for every curvature."""
    for c in CS:
        v, x = points(c)
        d0 = hyp.poincare_dist(torch.zeros(1, DIM, dtype=torch.float64), x, c)[0]
        assert close(d0, 2 * v.norm(dim=-1), 1e-6), c


def test_radius_equals_distance_to_origin():
    for c in CS:
        v, x = points(c)
        assert close(head(c).radius(x), 2 * v.norm(dim=-1), 1e-6), c


def test_dist_triangle_inequality():
    for c in CS:
        _, x = points(c)
        D = hyp.poincare_dist(x, x, c)
        worst = (D[:, None, :] - D[:, :, None] - D[None, :, :]).max()  # d(i,k) - d(i,j) - d(j,k)
        assert float(worst) < 1e-6, c


def test_dist_invariant_under_mobius_translation():
    """Left translation by a is an isometry: d(a(+)x, a(+)y) = d(x,y)."""
    for c in CS:
        _, x = points(c, 0)
        _, y = points(c, 1)
        _, a = points(c, 2)
        a0 = a[:1].expand_as(x)
        d1 = hyp.poincare_dist(hyp.mobius_add(a0, x, c), hyp.mobius_add(a0, y, c), c)
        assert close(d1, hyp.poincare_dist(x, y, c), 1e-6), c


def test_dist_small_curvature_limit():
    """As c -> 0 the ball becomes flat with metric factor 2: d(x,y) -> 2|x-y|."""
    c = 1e-6
    _, x = points(1.0, 0)
    _, y = points(1.0, 1)
    assert close(hyp.poincare_dist(x, y, c), 2 * torch.cdist(x, y), 1e-3)


def test_float32_agrees_with_float64():
    """Relative error of float32 distances stays below 1e-3 for tangent norms up to 1.5 (the trained heads have mean |v| = 1).
    It grows near the boundary: about 1e-3 at sqrt(c)|v| = 4."""
    for c in CS:
        _, x = points(c, 0)
        _, y = points(c, 1)
        d64 = hyp.poincare_dist(x, y, c)
        d32 = hyp.poincare_dist(x.float(), y.float(), c).double()
        assert float(((d32 - d64).abs() / d64.clamp_min(1e-9)).max()) < 1e-3, c


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f(); print(f"PASS  {n}")
        except AssertionError as e:
            failed += 1; print(f"FAIL  {n}  (c={e})")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
