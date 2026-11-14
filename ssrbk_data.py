# -*- coding: utf-8 -*-
"""Scenarios and datasets.

lorenz  : Benchmark 1 (Jung et al. Lorenz setup) + impulsive noise / initial mismatch options
nclt    : Benchmark 2 (Michigan NCLT, Wiener-velocity model) from a preprocessed .npz
doppler : Benchmark 3 surrogate (synthetic Doppler-line I/Q tracking); real SWellEx-96 /
          ShipsEar data need their own front-end (basebanding around the tonal) - see README.

Every sequence stores the analog pre-quantization measurement y = h(x) + v and the
unit-variance internal-noise samples eta (common random numbers for all 1-bit methods);
quantization itself happens inside each filter because the thresholds are closed-loop.
"""
import math
from dataclasses import dataclass

import numpy as np
import torch

from ssrbk_common import F64, psd_sqrt, external_noise, LorenzModel, WienerVelocity2D, DopplerModel


@dataclass
class SeqData:
    z: torch.Tensor     # [B, T, m] true states x_1..x_T
    y: torch.Tensor     # [B, T, n] analog measurements h(x_t) + v_t
    eta: torch.Tensor   # [B, T, n, N] unit-variance internal SSR noise
    z0: torch.Tensor    # [B, m] initial filter estimate (true x_0 + mismatch)
    P0d: torch.Tensor   # [B, m] diagonal of the initial filter covariance

    @property
    def B(self):
        return self.z.shape[0]

    @property
    def T(self):
        return self.z.shape[1]

    def slice(self, idx):
        return SeqData(self.z[idx], self.y[idx], self.eta[idx], self.z0[idx], self.P0d[idx])

    def to(self, device=None, dtype=None):
        return SeqData(*(x.to(device=device, dtype=dtype) for x in
                         (self.z, self.y, self.eta, self.z0, self.P0d)))


@dataclass
class Scenario:
    name: str
    model_true: object      # generates the data
    model_filt: object      # used by the filters (may be mismatched)
    eval_idx: list          # state components entering MSE / training loss
    sx: torch.Tensor        # per-component state std (for the initial mismatch)


def _rollout(model, x0, T, gen):
    L = psd_sqrt(model.Q)
    x, xs = x0, []
    for _ in range(T):
        x = model.f(x) + torch.randn(x.shape, generator=gen, dtype=F64) @ L.T
        xs.append(x)
    return torch.stack(xs, 1)


def _observe(model, xs):
    B, T, m = xs.shape
    return model.h(xs.reshape(B * T, m)).reshape(B, T, -1)


def _finalize(yc, x0, xs, arr, args, gen, sx, beta, r2, add_noise=True):
    B, T, m = xs.shape
    n = yc.shape[-1]
    if add_noise:
        y = yc + external_noise(args.noise, (B, T, n), gen, r2, alpha=args.alpha,
                                scale=args.noise_scale, eps=args.mix_eps,
                                kappa=args.mix_kappa, clip=args.clip)
    else:
        y = yc.clone()
    eta = arr.noise.sample((B, T, n, arr.N), gen, F64)
    sgn = (torch.rand(B, m, generator=gen, dtype=F64) < 0.5).to(F64) * 2.0 - 1.0
    dev = beta * sx / math.sqrt(m)                 # |mismatch| = beta * ||sx|| / sqrt(m) per comp.
    z0 = x0 + dev * sgn
    if args.P0 == 'informed':
        P0d = (dev ** 2 + args.P0_floor).expand(B, m).clone()
    else:
        P0d = torch.full((B, m), args.P0_small, dtype=F64)
    return SeqData(xs, y, eta, z0, P0d)


def _splits(args):
    return (('train', args.n_train, args.T_train, args.train_mismatch),
            ('cv', args.n_cv, args.T_train, args.train_mismatch),
            ('test', args.n_test, args.T_test, args.init_mismatch))


# ---------------------------------------------------------------------------
def _build_lorenz(args, arr, gen):
    mt = LorenzModel(dt=args.dt, J=args.J_data, q2=args.q2, r2=args.r2, rot_deg=args.rot_deg)
    mf = LorenzModel(dt=args.dt, J=args.J_model, q2=args.q2, r2=args.r2)
    raw = {}
    for key, B, T, beta in _splits(args):
        x = torch.tensor([1.0, 1.0, 1.0], dtype=F64) + 5.0 * torch.randn(B, 3, generator=gen, dtype=F64)
        for _ in range(args.burn):                 # land on the attractor
            x = mt.f(x)
        raw[key] = (x, _rollout(mt, x, T, gen), beta)
    sx = raw['train'][1].reshape(-1, 3).std(0)
    out = {k: _finalize(_observe(mt, xs), x0, xs, arr, args, gen, sx, beta, float(mt.R[0, 0]))
           for k, (x0, xs, beta) in raw.items()}
    return Scenario('lorenz', mt, mf, [0, 1, 2], sx), out['train'], out['cv'], out['test']


def _build_doppler(args, arr, gen):
    kw = dict(dt=args.dop_dt, amp=args.dop_amp, q_phi=args.dop_qphi, q_f=args.dop_qf,
              q_fd=args.dop_qfd, r2=args.dop_r2)
    mt, mf = DopplerModel(**kw), DopplerModel(**kw)
    sx = torch.tensor([math.pi / math.sqrt(3.0), args.dop_frange / math.sqrt(3.0),
                       max(args.dop_fdstd, 1e-3)], dtype=F64)
    out = {}
    for key, B, T, beta in _splits(args):
        x0 = torch.stack([2.0 * math.pi * torch.rand(B, generator=gen, dtype=F64),
                          args.dop_frange * (2.0 * torch.rand(B, generator=gen, dtype=F64) - 1.0),
                          args.dop_fdstd * torch.randn(B, generator=gen, dtype=F64)], -1)
        xs = _rollout(mt, x0, T, gen)
        out[key] = _finalize(_observe(mt, xs), x0, xs, arr, args, gen, sx, beta, float(mt.R[0, 0]))
    return Scenario('doppler', mt, mf, [1], sx), out['train'], out['cv'], out['test']


def _nclt_states(pos, dt):
    vel = np.gradient(pos, dt, axis=0)
    acc = np.gradient(vel, dt, axis=0)
    z = np.stack([pos[:, 0], vel[:, 0], acc[:, 0], pos[:, 1], vel[:, 1], acc[:, 1]], -1)
    return torch.tensor(z, dtype=F64)


def _build_nclt(args, arr, gen):
    if not args.nclt_npz:
        raise SystemExit('NCLT: pass --nclt_npz (create it with "python ssr_bknet.py prepare-nclt ...")')
    D = np.load(args.nclt_npz)
    dt = 1.0 / args.nclt_hz
    mt = WienerVelocity2D(dt=dt, q2=args.q2, r2=args.r2)
    mf = WienerVelocity2D(dt=dt, q2=args.q2, r2=args.r2)
    add = args.noise in ('sas', 'mix')     # real odometry is already noisy; only impulses are added
    r2 = float(args.r2)
    ztr = _nclt_states(D['train_pos'], dt)
    ytr = torch.tensor(D['train_vel'], dtype=F64)
    L = args.T_train
    K = (ztr.shape[0] - 1) // L
    zs = torch.stack([ztr[i * L + 1:(i + 1) * L + 1] for i in range(K)])
    ys = torch.stack([ytr[i * L + 1:(i + 1) * L + 1] for i in range(K)])
    x0 = torch.stack([ztr[i * L] for i in range(K)])
    perm = torch.randperm(K, generator=gen)
    ncv = max(1, int(round(0.15 * K)))
    sx = ztr.std(0)
    cv = _finalize(ys[perm[:ncv]], x0[perm[:ncv]], zs[perm[:ncv]], arr, args, gen, sx,
                   args.train_mismatch, r2, add)
    tr = _finalize(ys[perm[ncv:]], x0[perm[ncv:]], zs[perm[ncv:]], arr, args, gen, sx,
                   args.train_mismatch, r2, add)
    zte = _nclt_states(D['test_pos'], dt)
    yte = torch.tensor(D['test_vel'], dtype=F64)
    T = min(args.T_test, zte.shape[0] - 1)
    te = _finalize(yte[1:T + 1][None], zte[0][None], zte[1:T + 1][None], arr, args, gen, sx,
                   args.init_mismatch, r2, add)
    return Scenario('nclt', mt, mf, [0, 3], sx), tr, cv, te


def build_scenario(args, arr, gen):
    if args.scenario == 'lorenz':
        return _build_lorenz(args, arr, gen)
    if args.scenario == 'doppler':
        return _build_doppler(args, arr, gen)
    if args.scenario == 'nclt':
        return _build_nclt(args, arr, gen)
    raise ValueError(f'unknown scenario "{args.scenario}"')


# ---------------------------------------------------------------------------
def prepare_nclt(gt_csv, odom_csv, hz=1.0):
    """One possible NCLT preprocessing (check against your own pipeline):
    groundtruth_<date>.csv and odometry_mu_100hz.csv, columns [utime(us), x, y, ...].
    Both are resampled to `hz`; odometry positions are rigidly aligned to the ground-truth
    frame (2-D Kabsch over the whole run) and differenced to give velocity measurements."""
    gt = np.genfromtxt(gt_csv, delimiter=',')
    od = np.genfromtxt(odom_csv, delimiter=',')
    gt = gt[np.isfinite(gt[:, :3]).all(1)]
    od = od[np.isfinite(od[:, :3]).all(1)]
    gt, od = gt[np.argsort(gt[:, 0])], od[np.argsort(od[:, 0])]
    t0, t1 = max(gt[0, 0], od[0, 0]), min(gt[-1, 0], od[-1, 0])
    tg = np.arange(t0, t1, 1e6 / hz)
    G = np.stack([np.interp(tg, gt[:, 0], gt[:, 1]), np.interp(tg, gt[:, 0], gt[:, 2])], 1)
    O = np.stack([np.interp(tg, od[:, 0], od[:, 1]), np.interp(tg, od[:, 0], od[:, 2])], 1)
    Gc, Oc = G - G.mean(0), O - O.mean(0)
    U, _, Vt = np.linalg.svd(Oc.T @ Gc)
    Dm = np.diag([1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    Rm = Vt.T @ Dm @ U.T
    vel = np.vstack([np.zeros((1, 2)), np.diff(O, axis=0) * hz]) @ Rm.T
    vel[0] = vel[1]
    return G, vel
