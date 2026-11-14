# -*- coding: utf-8 -*-
"""Model-based recursive filters.

EKF            analog (full-precision) reference
BKF            Bussgang-aided KF / rBKF of Jung et al. (adaptive threshold, arcsine law),
               optionally with L comparators per feature and a FIXED internal dither
RLSEKFTracker  previous 1-bit RLS-EKF SSR quantizer (Wang et al., SPL letter): fixed thresholds,
               weights / noise scale calibrated by Eqs. (7), (9) on analog data, then an EKF tracker
SSRBKF         proposed generalized Bussgang-SSR KF with joint location-scale adaptation (Step 3)
"""
import math
from collections import deque

import torch

from ssrbk_common import sym, bmv, bdiag, qsign


class AnalyticFilter:
    name = 'analytic'

    def __init__(self, model):
        self.model = model

    def reset(self, z0, Sig0, data=None):
        self.z, self.S = z0.clone(), Sig0.clone()

    def predict(self):
        Fj = self.model.F(self.z)
        zp = self.model.f(self.z)
        Sp = sym(Fj @ self.S @ Fj.transpose(-1, -2) + self.model.Qm(zp))
        return zp, Sp, Fj

    def meas_prior(self, zp, Sp):
        H = self.model.H(zp)
        yp = self.model.h(zp)
        P = sym(H @ Sp @ H.transpose(-1, -2) + self.model.Rm(zp))
        return yp, H, P

    @staticmethod
    def gain(Sp, H, S):
        """K = Sp H^T S^-1 (Sp, S symmetric)."""
        return torch.linalg.solve(S, H @ Sp).transpose(-1, -2)

    def finish(self, zpost, Spost):
        m = Spost.shape[-1]
        self.z = zpost
        self.S = sym(Spost) + 1e-9 * torch.eye(m, dtype=Spost.dtype, device=Spost.device)
        return self.z


class EKF(AnalyticFilter):
    name = 'EKF (analog, full precision)'

    def step(self, t, y, eta):
        zp, Sp, _ = self.predict()
        yp, H, P = self.meas_prior(zp, Sp)
        K = self.gain(Sp, H, P)
        return self.finish(zp + bmv(K, y - yp), Sp - K @ P @ K.transpose(-1, -2))


class BKF(AnalyticFilter):
    """BKF (L=1, no dither) or rBKF with L branches sharing the adaptive threshold and an
    independent fixed internal dither sigma_d per branch (branch averaging, Eqs. 44-52)."""

    def __init__(self, model, L=1, dither=0.0):
        super().__init__(model)
        self.L, self.dither = int(L), float(dither)
        self.name = 'BKF (Jung et al.)' if (self.L == 1 and self.dither == 0.0) else \
            f'rBKF + fixed dither (L={self.L}, sigma_d={self.dither:.3g})'

    def step(self, t, y, eta):
        zp, Sp, _ = self.predict()
        yp, H, P = self.meas_prior(zp, Sp)
        e = y - yp                                             # threshold tau = yhat_{t|t-1}
        if self.L == 1 and self.dither == 0.0:
            r = qsign(e)
        else:
            r = qsign(e.unsqueeze(-1) + self.dither * eta[..., :self.L]).mean(-1)
        var = bdiag(P) + self.dither ** 2
        sd = var.sqrt()
        Bv = math.sqrt(2.0 / math.pi) / sd                     # Bussgang gain per branch (Eq. 29)
        rho = (P / (sd.unsqueeze(-1) * sd.unsqueeze(-2))).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        S = (2.0 / math.pi) * torch.asin(rho)                  # arcsine law (Eq. 38)
        rho_same = (bdiag(P) / var).clamp(max=1.0 - 1e-7)
        dS = 1.0 / self.L + (1.0 - 1.0 / self.L) * (2.0 / math.pi) * torch.asin(rho_same)
        S = S - torch.diag_embed(bdiag(S)) + torch.diag_embed(dS)   # reduced S* = A S A^T
        G = self.gain(Sp, Bv.unsqueeze(-1) * H, S)             # BG = Sigma (B H)^T S^-1 (Eq. 50)
        return self.finish(zp + bmv(G, r), Sp - G @ S @ G.transpose(-1, -2))


# ---------------------------------------------------------------------------
@torch.no_grad()
def calibrate_rlsekf(arr, y, eta, max_samples=20000, lam_w=0.1, P0=10.0,
                     Q_sig=1e-3, R_sig=1.0, Ps0=2.0, sig0=0.5):
    """RLS (Eq. 7) + scalar EKF on sigma (Eq. 9) of the SPL letter, supervised by the analog
    input as in the letter, with fixed thresholds theta_m = c + a d_m. Run in normalized
    units (y - c)/a (the letter's parameters assume unit variance; scale-equivariance makes
    this equivalent to raw units)."""
    dt = torch.float64
    n = y.shape[-1]
    Y = y.reshape(-1, n).to(dt)[:max_samples]
    E = eta.reshape(-1, n, eta.shape[-1]).to(dt)[:max_samples]
    c, a = Y.mean(0), Y.std(0).clamp_min(1e-9)
    X = (Y - c) / a
    d = arr.d.to(X)
    th = d[arr.gidx]
    G = arr.Gavg.to(X).transpose(0, 1)
    M = arr.M
    w = torch.full((n, M), 1.0 / M, dtype=dt)
    Pw = P0 * torch.eye(M, dtype=dt).repeat(n, 1, 1)
    sig = torch.full((n,), sig0, dtype=dt)
    Ps = torch.full((n,), Ps0, dtype=dt)
    K = X.shape[0]
    sq, cnt = torch.zeros(n, dtype=dt), 0
    for k in range(K):
        x = X[k]
        s = qsign(x.unsqueeze(-1) - th + sig.unsqueeze(-1) * E[k, :, :arr.N]) @ G     # [n, M]
        eps = x - (w * s).sum(-1)
        Pws = (Pw @ s.unsqueeze(-1)).squeeze(-1)
        g = Pws / ((s * Pws).sum(-1, keepdim=True) + lam_w)
        xd = x.unsqueeze(-1) - d
        dh = -2.0 * xd / sig.unsqueeze(-1) ** 2 * arr.noise.pdf(xd / sig.unsqueeze(-1))
        Hk = (w * dh).sum(-1)                                   # uses w_{k-1}
        Kk = Ps * Hk / (Hk * Hk * Ps + R_sig)
        sig = (sig + Kk * eps).clamp_min(1e-3)
        Ps = (1.0 - Kk * Hk) * Ps + Q_sig
        w = w + g * eps.unsqueeze(-1)
        Pw = sym(Pw - g.unsqueeze(-1) * Pws.unsqueeze(-2))
        if k >= K // 2:
            sq, cnt = sq + eps * eps, cnt + 1
    return dict(c=c, a=a, theta=c.unsqueeze(-1) + a.unsqueeze(-1) * d, sig=sig * a,
                w=w * a.unsqueeze(-1), Rdec=(sq / max(cnt, 1)) * a * a, sig_norm=sig, w_norm=w)


class RLSEKFTracker(AnalyticFilter):
    def __init__(self, model, arr, cal):
        super().__init__(model)
        self.arr, self.cal = arr, cal
        self.name = 'RLS-EKF (fixed thresholds, previous)'

    def step(self, t, y, eta):
        zp, Sp, _ = self.predict()
        yp, H, P = self.meas_prior(zp, Sp)
        cal, arr = self.cal, self.arr
        th = cal['theta'].to(y)[:, arr.gidx.to(y.device)]                       # [n, N]
        u = y.unsqueeze(-1) - th + cal['sig'].to(y).unsqueeze(-1) * eta[..., :arr.N]
        s = qsign(u) @ arr.Gavg.to(y).transpose(0, 1)
        ydec = cal['c'].to(y) + (cal['w'].to(y) * s).sum(-1)                    # decoded y
        P = sym(P + torch.diag_embed(cal['Rdec'].to(y)))
        K = self.gain(Sp, H, P)
        return self.finish(zp + bmv(K, ydec - yp), Sp - K @ P @ K.transpose(-1, -2))


# ---------------------------------------------------------------------------
class SSRBKF(AnalyticFilter):
    """Proposed generalized Bussgang-SSR Kalman filter (Steps 3.1-3.4).

    weight_mode : 'lmmse'  w = Cov(s)^-1 E[u s] from the known f_eta and u ~ N(0, v)
                  'rls'    robust M-RLS on a self-referential target (cfg.rls_target)
                  'uniform'
    kappa_mode  : 'opt' kappa* (Theorem 1), 'ekf' scalar EKF of the SPL letter on kappa, 'fixed'
    gain_mode   : 'stat'   Bussgang statistical linearization averaged over u ~ N(0, v)
                  'taylor' prior-free first-order gain at u = 0 (proposal's closed form)
    cfg.scale_ref: 'model' sqrt(P_jj) | 'robust' running median of |e~| | 'hybrid' max of both |
                   'const' fixed (location-only, 1-DoF ablation)
    """

    def __init__(self, model, arr, cfg, weight_mode='lmmse', kappa_mode='opt', gain_mode='stat',
                 name=None):
        super().__init__(model)
        self.arr, self.cfg = arr, cfg
        self.weight_mode, self.kappa_mode, self.gain_mode = weight_mode, kappa_mode, gain_mode
        self.kstar, self.zstar, self.wstar = arr.kappa_opt(1.0)
        self.name = name or f'SSR-BKF[{weight_mode},{kappa_mode},{gain_mode},{cfg.scale_ref}]'

    def reset(self, z0, Sig0, data=None):
        super().reset(z0, Sig0)
        cfg, M = self.cfg, self.arr.M
        B, n = z0.shape[0], self.model.n
        kw = dict(dtype=z0.dtype, device=z0.device)
        H0, R = self.model.H(z0), self.model.Rm(z0)
        s0 = bdiag(sym(H0 @ Sig0 @ H0.transpose(-1, -2) + R)).sqrt()
        self.Rd = bdiag(R)
        self.s_const = torch.full((n,), cfg.s_const, **kw) if cfg.s_const > 0 else self.Rd.sqrt()
        self.m_rob = 0.6745 * s0                      # running median of |e~| (robust scale)
        self.s_used = None
        k0 = cfg.kappa if self.kappa_mode == 'fixed' else self.kstar
        self.kappa = torch.full((B, n), k0, **kw)
        self.Pk = torch.full((B, n), cfg.P_kappa0, **kw)
        w0 = self.wstar.to(z0) if cfg.rls_init == 'lmmse' else torch.full((M,), 1.0 / M, **kw)
        self.w = w0.expand(B, n, M).clone()
        self.Pw = cfg.rls_P0 * torch.eye(M, **kw).expand(B, n, M, M).clone()
        self.m_eps = torch.full((B, n), 0.5, **kw)
        self.q_var = torch.zeros(B, n, **kw)          # quantization-noise variance of e~ (last step)
        self.hist = deque(maxlen=max(int(cfg.lag), 1) + 1)
        self.truth = data.z if data is not None else None

    def _scale_reference(self, Pd):
        ref = self.cfg.scale_ref
        s_rob = self.m_rob / 0.6745
        if ref == 'model':
            return Pd.sqrt()
        if ref == 'robust':
            return s_rob
        if ref == 'hybrid':
            return torch.maximum(Pd.sqrt(), s_rob)
        return self.s_const.expand_as(Pd)

    def step(self, t, y, eta):
        cfg, arr = self.cfg, self.arr
        zp, Sp, Fj = self.predict()
        yp, H, P = self.meas_prior(zp, Sp)
        if cfg.inflate and cfg.scale_ref in ('robust', 'hybrid'):
            # optional covariance matching: inflate Sigma_pred when the robust scale of e~ exceeds the
            # predicted one (R and the previous step's quantization-noise variance are subtracted)
            HSH = (bdiag(P) - self.Rd).clamp_min(1e-12)
            excess = (self.m_rob / 0.6745) ** 2 - self.Rd - self.q_var
            lam = (excess / HSH).clamp_min(1.0).amax(-1)
            Sp = Sp * lam[:, None, None]
            yp, H, P = self.meas_prior(zp, Sp)
        Pd = bdiag(P)
        # ---- Step 3.1: location (threshold centre yp) and scale (aperture) control ----
        s_ref = self._scale_reference(Pd)
        self.s_used = s_ref.clone() if self.s_used is None else \
            self.s_used + cfg.mu_eta * (s_ref - self.s_used)
        scale = self.s_used.clamp_min(1e-9)
        kappa = self.kappa
        sg = arr.bits(y, yp, scale, kappa, eta)                                  # [B, n, M]
        # ---- Step 2: generalized Bussgang-SSR quantities (normalized units u = e/scale) ----
        v = (Pd / scale ** 2).clamp(1e-6, 25.0)
        if self.weight_mode == 'lmmse' or self.gain_mode == 'stat':
            mu, p, C = arr.moments(v, kappa)
        if self.weight_mode == 'lmmse':
            w = arr.lmmse_weights(p, C)
        elif self.weight_mode == 'rls':
            w = self.w
        else:
            w = torch.full_like(sg, 1.0 / arr.M)
        if self.gain_mode == 'stat':
            gam, mean, D = arr.bussgang(w, mu, p, C, v)
        else:
            gam, mean, D = arr.taylor(w, kappa)
        gam = gam.clamp_min(1e-4)
        # ---- Step 3.2: spatial decoding -> unbiased residual pseudo-observation ----
        e_t = scale * ((w * sg).sum(-1) - mean) / gam
        # ---- Step 3.3: Bussgang-SSR Kalman update: e~ = H(z - zp) + v + xi/gamma ----
        self.q_var = scale ** 2 * D / gam ** 2
        S = sym(P + torch.diag_embed(self.q_var))
        K = self.gain(Sp, H, S)
        self.finish(zp + bmv(K, e_t), Sp - K @ S @ K.transpose(-1, -2))
        # ---- Step 3.4: feedback (robust scale, M-RLS weights, kappa) ----
        self._update_scale(e_t, sg)
        self.hist.append(dict(zp=zp, Sp=Sp, F=Fj, z=self.z, S=self.S, sg=sg, scale=scale,
                              kappa=kappa.clone(), w=w.clone(), yp=yp))
        tgt = self._target(t)
        if tgt is not None:
            ent, d = tgt
            u_t = d / ent['scale']
            if self.weight_mode == 'rls':
                self._rls(ent['sg'], u_t)
            if self.kappa_mode == 'ekf':
                self._kappa_ekf(ent, u_t)
        return self.z

    def _update_scale(self, e_t, sg):
        """Multiplicative stochastic-approximation median of |e~| plus fast opening on
        full saturation (all comparators agree => residual outside the capture range)."""
        cfg = self.cfg
        ind = (e_t.abs() > self.m_rob).to(e_t.dtype)
        sat = (sg.mean(-1).abs() >= 1.0 - 1e-9).to(e_t.dtype)
        self.m_rob = (self.m_rob * torch.exp(cfg.mu_s * (ind - 0.5) + cfg.mu_sat * sat)).clamp(1e-9, 1e9)

    def _target(self, t):
        """Noise-free residual target d = h(z_target) - yhat for the RLS / kappa-EKF feedback."""
        if self.weight_mode != 'rls' and self.kappa_mode != 'ekf':
            return None
        mode = self.cfg.rls_target
        if mode == 'posterior':            # proposal as written: s_{t|t} - s_{t|t-1}
            ent = self.hist[-1]
            return ent, self.model.h(ent['z']) - ent['yp']
        if mode == 'oracle':               # diagnostic upper bound only (uses the true state)
            ent = self.hist[-1]
            return ent, self.model.h(self.truth[:, t].to(ent['z'])) - ent['yp']
        if len(self.hist) < self.hist.maxlen:
            return None
        zs = self.hist[-1]['z']            # fixed-lag RTS smoother: z_{t-L | t}
        for k in range(len(self.hist) - 2, -1, -1):
            a, b = self.hist[k], self.hist[k + 1]
            Ck = torch.linalg.solve(b['Sp'], b['F'] @ a['S']).transpose(-1, -2)
            zs = a['z'] + bmv(Ck, zs - b['zp'])
        ent = self.hist[0]
        return ent, self.model.h(zs) - ent['yp']

    def _rls(self, sg, u_t):
        """Robust (Huber-weighted) version of the letter's RLS, Eq. (7), with exponential
        forgetting P <- P / lambda_f; q = 1 recovers Eq. (7) exactly. Normalized units."""
        cfg = self.cfg
        eps = u_t - (self.w * sg).sum(-1)
        a = eps.abs() / (self.m_eps / 0.6745).clamp_min(1e-9)
        q = torch.where(a <= cfg.huber, torch.ones_like(a), cfg.huber / a.clamp_min(1e-12))
        Ps = (self.Pw @ sg.unsqueeze(-1)).squeeze(-1)
        den = cfg.lam_w + q * (sg * Ps).sum(-1)
        g = (q / den).unsqueeze(-1) * Ps
        self.w = self.w + g * eps.unsqueeze(-1)
        self.Pw = sym(self.Pw - g.unsqueeze(-1) * Ps.unsqueeze(-2)) / cfg.lam_f
        self.m_eps = self.m_eps * torch.exp(cfg.mu_s * ((eps.abs() > self.m_eps).to(eps.dtype) - 0.5))

    def _kappa_ekf(self, ent, u_t):
        """Scalar EKF of the SPL letter (Eq. 9) on the dimensionless aperture kappa."""
        cfg, arr = self.cfg, self.arr
        k_used, w_used = ent['kappa'], ent['w']
        xd = u_t.unsqueeze(-1) - arr.d.to(u_t)
        dh = -2.0 * xd / k_used.unsqueeze(-1) ** 2 * arr.noise.pdf(xd / k_used.unsqueeze(-1))
        Hk = (w_used * dh).sum(-1)
        eps = u_t - (w_used * ent['sg']).sum(-1)
        Kk = self.Pk * Hk / (Hk * Hk * self.Pk + cfg.R_kappa)
        self.kappa = (self.kappa + Kk * eps).clamp(cfg.kappa_min, cfg.kappa_max)
        self.Pk = (1.0 - Kk * Hk) * self.Pk + cfg.Q_kappa


@torch.no_grad()
def run_analytic(filt, data):
    """Runs a model-based filter on CPU in float64; returns x_hat [B, T, m]."""
    dt = torch.float64
    y, eta = data.y.to(dt), data.eta.to(dt)
    filt.reset(data.z0.to(dt), torch.diag_embed(data.P0d.to(dt)), data)
    out = [filt.step(t, y[:, t], eta[:, t]) for t in range(y.shape[1])]
    return torch.stack(out, 1)
