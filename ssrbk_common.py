# -*- coding: utf-8 -*-
"""Shared utilities, noise models and state-space models.

Part of the SSR-Bussgang-Kalman package; the entry point is ssr_bknet.py.
"""
import math

import torch

F64 = torch.float64


# ---------------------------------------------------------------------------
# batched linear-algebra helpers (last two dims are matrices)
# ---------------------------------------------------------------------------
def sym(A):
    return 0.5 * (A + A.transpose(-1, -2))


def bmv(A, x):
    """A [..., a, b] times x [..., b] -> [..., a]."""
    return (A @ x.unsqueeze(-1)).squeeze(-1)


def bdiag(A):
    return torch.diagonal(A, dim1=-2, dim2=-1)


def qsign(u):
    """1-bit comparator q(u) of Jung et al. (Eq. 14): +1 if u > 0, -1 otherwise."""
    return torch.where(u > 0, torch.ones_like(u), -torch.ones_like(u))


def psd_sqrt(Q):
    """L with L @ L.T == Q for a symmetric PSD (possibly singular) matrix."""
    lam, V = torch.linalg.eigh(sym(Q))
    return V * lam.clamp_min(0.0).sqrt().unsqueeze(-2)


def to_db(x):
    x = float(x)
    if math.isnan(x):
        return float('nan')
    if math.isinf(x):
        return float('inf')
    return 10.0 * math.log10(max(x, 1e-30))


def batch_jacobian(fn, z):
    """Jacobian of a row-wise map fn: [B, m] -> [B, k], returned as [B, k, m].

    k reverse-mode passes; also works inside torch.no_grad()."""
    with torch.enable_grad():
        zz = z.detach().clone().requires_grad_(True)
        out = fn(zz)
        k = out.shape[-1]
        if not out.requires_grad:
            return torch.zeros(z.shape[0], k, z.shape[1], dtype=z.dtype, device=z.device)
        rows = []
        for i in range(k):
            g = torch.autograd.grad(out[:, i].sum(), zz, retain_graph=(i < k - 1),
                                    allow_unused=True)[0]
            rows.append(torch.zeros_like(zz) if g is None else g)
    return torch.stack(rows, dim=1).detach()


# ---------------------------------------------------------------------------
# internal (injected, known) SSR noise: unit variance, symmetric, scale family
# ---------------------------------------------------------------------------
class InternalNoise:
    """Unit-variance symmetric internal noise eta with pdf f_eta and cdf F_eta.
    The injected noise is sigma_eta * eta, i.e. density f_eta(x / sigma_eta) / sigma_eta."""
    KINDS = ('gauss', 'logistic', 'uniform', 'laplace')

    def __init__(self, kind='gauss'):
        kind = kind.lower()
        if kind not in self.KINDS:
            raise ValueError(f'unknown internal noise "{kind}", choose from {self.KINDS}')
        self.kind = kind

    def cdf(self, x):
        k = self.kind
        if k == 'gauss':
            return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        if k == 'logistic':
            return torch.sigmoid(x * (math.pi / math.sqrt(3.0)))
        if k == 'uniform':
            a = math.sqrt(3.0)
            return ((x + a) / (2.0 * a)).clamp(0.0, 1.0)
        b = 1.0 / math.sqrt(2.0)
        return 0.5 + 0.5 * torch.sign(x) * (1.0 - torch.exp(-x.abs() / b))

    def pdf(self, x):
        k = self.kind
        if k == 'gauss':
            return torch.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)
        if k == 'logistic':
            s = math.sqrt(3.0) / math.pi
            p = torch.sigmoid(x / s)
            return p * (1.0 - p) / s
        if k == 'uniform':
            a = math.sqrt(3.0)
            return (x.abs() < a).to(x.dtype) / (2.0 * a)
        b = 1.0 / math.sqrt(2.0)
        return torch.exp(-x.abs() / b) / (2.0 * b)

    def sample(self, shape, gen, dtype=F64):
        k = self.kind
        if k == 'gauss':
            return torch.randn(shape, generator=gen, dtype=dtype)
        U = torch.rand(shape, generator=gen, dtype=dtype).clamp(1e-7, 1.0 - 1e-7)
        if k == 'logistic':
            return (math.sqrt(3.0) / math.pi) * torch.log(U / (1.0 - U))
        if k == 'uniform':
            return math.sqrt(3.0) * (2.0 * U - 1.0)
        U2 = torch.rand(shape, generator=gen, dtype=dtype).clamp(1e-7, 1.0 - 1e-7)
        return (torch.log(U) - torch.log(U2)) / math.sqrt(2.0)


# ---------------------------------------------------------------------------
# external (unknown, environmental) noise v_t, common to all comparators of a feature
# ---------------------------------------------------------------------------
def external_noise(kind, shape, gen, r2, alpha=1.5, scale=None, eps=0.05, kappa=10.0, clip=1e4):
    """gauss : N(0, r2)
    sas   : symmetric alpha-stable (Chambers-Mallows-Stuck), char. function exp(-|c t|^alpha);
            c defaults to sqrt(r2 / 2) so that alpha = 2 reproduces N(0, r2)
    mix   : (1 - eps) N(0, r2) + eps N(0, kappa^2 r2)  (epsilon-contaminated Gaussian)
    none  : zero
    Samples are clipped to +-clip only to keep float arithmetic finite."""
    r = math.sqrt(max(float(r2), 0.0))
    if kind == 'none':
        return torch.zeros(shape, dtype=F64)
    if kind == 'gauss':
        v = r * torch.randn(shape, generator=gen, dtype=F64)
    elif kind == 'sas':
        c = r / math.sqrt(2.0) if (scale is None or scale <= 0) else float(scale)
        U = (torch.rand(shape, generator=gen, dtype=F64) - 0.5) * math.pi
        U = U.clamp(-math.pi / 2 + 1e-9, math.pi / 2 - 1e-9)
        W = -torch.log(torch.rand(shape, generator=gen, dtype=F64).clamp_min(1e-300))
        if abs(alpha - 1.0) < 1e-9:
            X = torch.tan(U)
        else:
            X = (torch.sin(alpha * U) / torch.cos(U).pow(1.0 / alpha)
                 * (torch.cos((1.0 - alpha) * U) / W).pow((1.0 - alpha) / alpha))
        v = c * X
    elif kind == 'mix':
        z = torch.randn(shape, generator=gen, dtype=F64)
        out = (torch.rand(shape, generator=gen, dtype=F64) < eps).to(F64)
        v = r * z * (1.0 + (kappa - 1.0) * out)
    else:
        raise ValueError(f'unknown external noise "{kind}"')
    return v.clamp(-clip, clip)


# ---------------------------------------------------------------------------
# state-space models: x_t = f(x_{t-1}) + w_t,  y_t = h(x_t) + v_t  (row-wise batched)
# ---------------------------------------------------------------------------
class SSModel:
    m = 0
    n = 0
    Q = None
    R = None

    def f(self, z):
        raise NotImplementedError

    def h(self, z):
        raise NotImplementedError

    def F(self, z):
        return batch_jacobian(self.f, z)

    def H(self, z):
        return batch_jacobian(self.h, z)

    def Qm(self, ref):
        return self.Q.to(ref)

    def Rm(self, ref):
        return self.R.to(ref)


class LorenzModel(SSModel):
    """Lorenz attractor in state-dependent-coefficient form (Jung et al., Eqs. 59-61):
    f_J(x) = Phi_J(x) x,  Phi_J = I + sum_{j=1..J} (A(x) dt)^j / j!,  h(x) = Rot x (Rot = I nominally)."""

    def __init__(self, dt=0.02, J=5, q2=1e-3, r2=1e-1, rot_deg=0.0,
                 sig=10.0, rho=28.0, beta=8.0 / 3.0):
        self.m, self.n = 3, 3
        self.dt, self.J = float(dt), int(J)
        self.sig, self.rho, self.beta = float(sig), float(rho), float(beta)
        self.Q = q2 * torch.eye(3, dtype=F64)
        self.R = r2 * torch.eye(3, dtype=F64)
        a = math.radians(rot_deg)
        self.rotated = abs(rot_deg) > 0.0
        self.Hrot = torch.tensor([[math.cos(a), -math.sin(a), 0.0],
                                  [math.sin(a), math.cos(a), 0.0],
                                  [0.0, 0.0, 1.0]], dtype=F64)

    def f(self, x):
        one = torch.ones_like(x[:, 0])
        zero = torch.zeros_like(one)
        A = torch.stack([
            torch.stack([-self.sig * one, self.sig * one, zero], -1),
            torch.stack([self.rho - x[:, 2], -one, zero], -1),
            torch.stack([x[:, 1], zero, -self.beta * one], -1)], -2)
        Md = A * self.dt
        eye = torch.eye(3, dtype=x.dtype, device=x.device).expand_as(Md)
        Phi, term = eye.clone(), eye.clone()
        for j in range(1, self.J + 1):
            term = term @ Md / j
            Phi = Phi + term
        return bmv(Phi, x)

    def h(self, x):
        return x @ self.Hrot.to(x).T if self.rotated else x

    def H(self, x):
        Hm = self.Hrot if self.rotated else torch.eye(3, dtype=F64)
        return Hm.to(x).expand(x.shape[0], 3, 3)


class WienerVelocity2D(SSModel):
    """NCLT model of Jung et al. (Eqs. 72-75): per axis [p, v, a]; odometry velocities are measured."""

    def __init__(self, dt=1.0, q2=1e-2, r2=1e-2):
        self.m, self.n = 6, 2
        Fa = torch.tensor([[1.0, dt, 0.5 * dt * dt], [0.0, 1.0, dt], [0.0, 0.0, 1.0]], dtype=F64)
        Qa = q2 * torch.tensor([[dt ** 4 / 4, dt ** 3 / 2, dt ** 2 / 2],
                                [dt ** 3 / 2, dt ** 2, dt],
                                [dt ** 2 / 2, dt, 1.0]], dtype=F64)
        self.Fm = torch.block_diag(Fa, Fa)
        self.Q = torch.block_diag(Qa, Qa)
        self.Hm = torch.zeros(2, 6, dtype=F64)
        self.Hm[0, 1] = 1.0
        self.Hm[1, 4] = 1.0
        self.R = r2 * torch.eye(2, dtype=F64)

    def f(self, z):
        return z @ self.Fm.to(z).T

    def h(self, z):
        return z @ self.Hm.to(z).T

    def F(self, z):
        return self.Fm.to(z).expand(z.shape[0], 6, 6)

    def H(self, z):
        return self.Hm.to(z).expand(z.shape[0], 2, 6)


class DopplerModel(SSModel):
    """Tonal / Doppler-line tracking surrogate (Benchmark 3):
    z = [phase (rad), frequency (Hz), frequency rate (Hz/s)],  y = A [cos(phase), sin(phase)] + v
    (baseband I/Q of a narrowband line, sampled every dt seconds)."""

    def __init__(self, dt=0.01, amp=1.0, q_phi=1e-4, q_f=1e-6, q_fd=1e-4, r2=1.0):
        self.m, self.n = 3, 2
        self.amp = float(amp)
        self.Fm = torch.tensor([[1.0, 2.0 * math.pi * dt, math.pi * dt * dt],
                                [0.0, 1.0, dt],
                                [0.0, 0.0, 1.0]], dtype=F64)
        self.Q = torch.diag(torch.tensor([q_phi, q_f, q_fd], dtype=F64))
        self.R = r2 * torch.eye(2, dtype=F64)

    def f(self, z):
        return z @ self.Fm.to(z).T

    def F(self, z):
        return self.Fm.to(z).expand(z.shape[0], 3, 3)

    def h(self, z):
        return self.amp * torch.stack([torch.cos(z[:, 0]), torch.sin(z[:, 0])], -1)

    def H(self, z):
        s, c = torch.sin(z[:, 0]), torch.cos(z[:, 0])
        zero = torch.zeros_like(s)
        return self.amp * torch.stack([torch.stack([-s, zero, zero], -1),
                                       torch.stack([c, zero, zero], -1)], -2)
