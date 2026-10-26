"""Target-shift geometry for the AR-shift trials (PHASE35_AR_PLANT.md, trials 2-4).

A shifted target t keeps cosine C with the true activation a and moves toward a reference vector u:
    t = C * a_hat + sqrt(1 - C^2) * w_hat,   w = u - (u . a_hat) a_hat
All vectors are compared in the AR's normalised space, so only directions matter.
"""
import math
import torch

C = 0.95


def unit(x):
    x = x.float()
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-12)


def shift_toward(a, u, c=C):
    """Rows of a (n, d) moved toward rows of u (n, d) or one vector u (d,) so that cos(t, a) = c. Unit-norm output."""
    ah = unit(a)
    u = u.float().expand_as(ah)
    w = u - (u * ah).sum(-1, keepdim=True) * ah
    return c * ah + math.sqrt(1 - c * c) * unit(w)


def random_dirs(k, d, seed):
    """k orthonormal random directions in R^d (rows)."""
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return q.T.contiguous()


def shift_coefs(p, a, U):
    """Coefficients (n, k) of predictions p (n, d) on the basis U (k, d) after removing the target direction a (n, d)
    from both; a and p are unit rows. The norm of a row, over sqrt(1 - C^2), is the fraction of a full shift present."""
    p, a, U = p.float(), a.float(), U.float().to(p.device)
    Up = U.unsqueeze(0) - (a @ U.T).unsqueeze(-1) * a.unsqueeze(1)          # (n, k, d)
    Up = Up / Up.norm(dim=-1, keepdim=True).clamp_min(1e-9)
    pp = p - (p * a).sum(-1, keepdim=True) * a                               # (n, d)
    return torch.einsum("nkd,nd->nk", Up, pp)


def shift_frac(p, a, U, c=C, chunk=256):
    """Fraction of a full target shift present in each row of p (see shift_coefs)."""
    e = math.sqrt(1 - c * c)
    return torch.cat([shift_coefs(p[i:i + chunk], a[i:i + chunk], U).norm(dim=-1) / e for i in range(0, len(p), chunk)])


def basis(vs):
    """Orthonormal basis (rows) of the span of vs (k, d)."""
    q, _ = torch.linalg.qr(vs.float().T)
    return q.T.contiguous()
