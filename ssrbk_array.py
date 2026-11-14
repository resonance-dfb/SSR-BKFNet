# -*- coding: utf-8 -*-
"""M-group SSR comparator array with location-scale control and the generalized
Bussgang-SSR moments (Steps 1-2 of the proposal).

Front-end for physical feature j, comparator n of group m:
    b_n = q( y - tau_m + kappa * s_ref * eta_n ),     tau_m = yhat_{t|t-1} + d_m * s_ref
    group averages  s_m = (1/N_m) sum_{n in m} b_n
- yhat_{t|t-1}      : dynamic threshold centre (location; the BKNet idea)
- delta_m = d_m s_ref : fixed hierarchical offsets
- sigma_eta = kappa s_ref : internal-noise scale (aperture)
Offsets and noise scale are both expressed in units of the residual-scale reference
s_ref. In normalized units u = (y - yhat)/s_ref the array is then scale-equivariant and the
optimal "resonance factor" kappa_opt is a single dimensionless constant (Theorem 1 of the
SPL letter, generalized to M groups and any f_eta).

All moments are conditional on the known internal noise only (exact for any f_eta) and are
averaged over the predicted residual u ~ N(0, v), v = P_jj / s_ref^2 (statistical / Bussgang
linearization). Quadrature: dense trapezoid grid weighted by the Gaussian density.
"""
import math

import torch

from ssrbk_common import F64, InternalNoise, qsign


class SSRArray:
    def __init__(self, offsets, per_group, noise_kind='gauss', n_quad=401, quad_halfwidth=8.0):
        self.d = torch.as_tensor(offsets, dtype=F64).reshape(-1)
        self.M = int(self.d.numel())
        self.per_group = int(per_group)
        self.N = self.M * self.per_group
        self.Nm = torch.full((self.M,), float(self.per_group), dtype=F64)
        self.gidx = torch.arange(self.N) // self.per_group
        G = torch.zeros(self.M, self.N, dtype=F64)
        G[self.gidx, torch.arange(self.N)] = 1.0 / self.per_group
        self.Gavg = G
        self.noise = InternalNoise(noise_kind)
        x = torch.linspace(-quad_halfwidth, quad_halfwidth, int(n_quad), dtype=F64)
        w = torch.exp(-0.5 * x * x)
        self.qx, self.qw = x, w / w.sum()

    # ----------------------------------------------------------------- front-end
    def bits(self, y, center, scale, kappa, eta):
        """y, center, scale, kappa: [B, n]; eta: [B, n, >=N] unit-variance samples.
        Returns the group averages s: [B, n, M] (values in [-1, 1])."""
        tau = center.unsqueeze(-1) + self.d.to(y) * scale.unsqueeze(-1)          # [B, n, M]
        g = self.gidx.to(y.device)
        u = y.unsqueeze(-1) - tau[..., g] + (kappa * scale).unsqueeze(-1) * eta[..., :self.N]
        return qsign(u) @ self.Gavg.to(y).transpose(0, 1)

    def mean_response(self, u, kappa):
        """E[s_m | u] = 2 F_eta((u - d_m) / kappa) - 1 in normalized units; output [..., M]."""
        return 2.0 * self.noise.cdf((u.unsqueeze(-1) - self.d.to(u)) / kappa.unsqueeze(-1)) - 1.0

    # ----------------------------------------------------- generalized Bussgang
    def moments(self, v, kappa):
        """For u ~ N(0, v) (v, kappa: same shape [...]) returns
        mu [..., M] = E[s],  p [..., M] = E[u s],  C [..., M, M] = Cov(s)."""
        x, w = self.qx.to(v), self.qw.to(v)
        u = v.clamp_min(1e-12).sqrt().unsqueeze(-1) * x                         # [..., K]
        G = self.mean_response(u, kappa.unsqueeze(-1))                            # [..., K, M]
        wG = w.unsqueeze(-1) * G
        mu = wG.sum(-2)
        p = (wG * u.unsqueeze(-1)).sum(-2)
        Psi = wG.transpose(-1, -2) @ G                                           # E[g_m g_l]
        var_q = (w.unsqueeze(-1) * (1.0 - G * G)).sum(-2) / self.Nm.to(v)       # E[Var(s_m|u)]
        C = Psi - mu.unsqueeze(-1) * mu.unsqueeze(-2) + torch.diag_embed(var_q)
        return mu, p, C

    @staticmethod
    def lmmse_weights(p, C, ridge=1e-10):
        """Spatial LMMSE decoder w = Cov(s)^-1 E[u s] (the SPL letter's w^o = R^-1 p)."""
        eye = torch.eye(C.shape[-1], dtype=C.dtype, device=C.device)
        return torch.linalg.solve(C + ridge * eye, p.unsqueeze(-1)).squeeze(-1)

    @staticmethod
    def bussgang(w, mu, p, C, v):
        """Generalized Bussgang decomposition  w^T s = w^T mu + gamma u + xi,  E[xi u] = 0.
        Returns gamma, mean = w^T mu, and the distortion variance D = Var(xi) (normalized units)."""
        gamma = (w * p).sum(-1) / v.clamp_min(1e-12)
        mean = (w * mu).sum(-1)
        Vw = (w.unsqueeze(-2) @ C @ w.unsqueeze(-1)).squeeze(-1).squeeze(-1)
        D = (Vw - gamma ** 2 * v).clamp_min(1e-12)
        return gamma, mean, D

    def taylor(self, w, kappa):
        """Prior-free first-order quantities of the proposal (Step 2), evaluated at u = 0:
        gamma = (2/kappa) sum_m w_m f_eta(d_m/kappa),  R_q = sum_m w_m^2 [1 - g_m(0)^2] / N_m."""
        z = -self.d.to(w) / kappa.unsqueeze(-1)
        G0 = 2.0 * self.noise.cdf(z) - 1.0
        gamma = (w * (2.0 / kappa.unsqueeze(-1)) * self.noise.pdf(z)).sum(-1)
        mean = (w * G0).sum(-1)
        Rq = (w * w * (1.0 - G0 * G0) / self.Nm.to(w)).sum(-1)
        return gamma, mean, Rq

    # ------------------------------------------------- Theorem 1 (generalized)
    def zeta(self, v, kappa):
        """Minimum decoding MSE zeta_min = v - p^T Cov(s)^-1 p and the LMMSE weights."""
        mu, p, C = self.moments(v, kappa)
        w = self.lmmse_weights(p, C)
        return v - (w * p).sum(-1), w

    def kappa_opt(self, v=1.0, lo=0.05, hi=5.0, n=300):
        """kappa* = argmin_kappa zeta_min(kappa) for u ~ N(0, v): grid search + golden section."""
        grid = torch.exp(torch.linspace(math.log(lo), math.log(hi), n, dtype=F64))
        z, _ = self.zeta(torch.full_like(grid, v), grid)
        i = int(torch.argmin(z))
        a, b = float(grid[max(i - 1, 0)]), float(grid[min(i + 1, n - 1)])

        def fz(k):
            zz, _ = self.zeta(torch.tensor([v], dtype=F64), torch.tensor([k], dtype=F64))
            return float(zz[0])

        phi = (math.sqrt(5.0) - 1.0) / 2.0
        c, d = b - phi * (b - a), a + phi * (b - a)
        fc, fd = fz(c), fz(d)
        for _ in range(50):
            if fc < fd:
                b, d, fd = d, c, fc
                c = b - phi * (b - a)
                fc = fz(c)
            else:
                a, c, fc = c, d, fd
                d = a + phi * (b - a)
                fd = fz(d)
        k = 0.5 * (a + b)
        zz, w = self.zeta(torch.tensor([v], dtype=F64), torch.tensor([k], dtype=F64))
        return k, float(zz[0]), w[0]
