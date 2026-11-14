#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Generalized Bussgang-SSR Kalman filtering with location-scale (threshold-noise) adaptation.

    python ssr_bknet.py selftest
    python ssr_bknet.py run                                    # Lorenz, nominal (Jung et al.)
    python ssr_bknet.py run --noise sas --alpha 1.5 --init_mismatch 2 --P0 small
    python ssr_bknet.py run --scenario doppler --noise sas --alpha 1.2 --dop_r2 2
    python ssr_bknet.py prepare-nclt --gt_train G1.csv --odom_train O1.csv \
                                     --gt_test G2.csv --odom_test O2.csv --out nclt.npz
    python ssr_bknet.py run --scenario nclt --nclt_npz nclt.npz --T_train 50
See README.md for the method keys and the mapping to the proposal.
"""
import argparse
import json
import math
import os
import tempfile
import time
import traceback

import numpy as np
import torch

from ssrbk_common import F64, LorenzModel, bdiag, external_noise, to_db
from ssrbk_array import SSRArray
from ssrbk_data import build_scenario, prepare_nclt
from ssrbk_filters import EKF, BKF, RLSEKFTracker, SSRBKF, calibrate_rlsekf, run_analytic
from ssrbk_learned import LearnedFilter, train_learned, run_learned

ANALYTIC = ('ekf', 'bkf', 'bkf_dither', 'rlsekf', 'ssrbkf', 'ssrbkf_1dof', 'ssrbkf_rls', 'ssrbkf_taylor')
LEARNED = ('knet', 'bknet', 'ssrbknet')
DEFAULT_METHODS = ','.join(ANALYTIC + LEARNED)


# ---------------------------------------------------------------------------
def build_analytic(name, scen, arr, args, cal):
    mf = scen.model_filt
    if name == 'ekf':
        return EKF(mf)
    if name == 'bkf':
        return BKF(mf, L=1, dither=0.0)
    if name == 'bkf_dither':
        return BKF(mf, L=arr.N, dither=args.dither)
    if name == 'rlsekf':
        return RLSEKFTracker(mf, arr, cal)
    cfg = argparse.Namespace(**vars(args))
    if name == 'ssrbkf':
        return SSRBKF(mf, arr, cfg, 'lmmse', args.kappa_mode, 'stat', 'SSR-BKF (proposed, 2-DoF)')
    if name == 'ssrbkf_1dof':
        cfg.scale_ref = 'const'
        return SSRBKF(mf, arr, cfg, 'lmmse', 'opt', 'stat', 'SSR-BKF 1-DoF (location only)')
    if name == 'ssrbkf_rls':
        return SSRBKF(mf, arr, cfg, 'rls', 'ekf', 'stat',
                      f'SSR-BKF + M-RLS/kappa-EKF ({args.rls_target} target)')
    if name == 'ssrbkf_taylor':
        return SSRBKF(mf, arr, cfg, 'lmmse', args.kappa_mode, 'taylor', 'SSR-BKF (Taylor gain)')
    raise ValueError(name)


def compute_metrics(xh, z, idx, tr_window):
    """MSE as in Jung et al. Eq. (62): squared error norm over idx, averaged over time and sequences."""
    e = ((xh[..., idx] - z[..., idx].to(xh)) ** 2).sum(-1)
    nonfinite = int((~torch.isfinite(e)).sum())
    e = torch.nan_to_num(e, nan=1e12, posinf=1e12, neginf=1e12).clamp(max=1e12)
    T = e.shape[1]
    tw = min(tr_window, T)
    seq_db = 10.0 * torch.log10(e.mean(1).clamp_min(1e-30))
    return dict(mse_db=to_db(e.mean()), seq_db_mean=float(seq_db.mean()),
                seq_db_std=float(seq_db.std()) if e.shape[0] > 1 else 0.0,
                transient_db=to_db(e[:, :tw].mean()),
                steady_db=to_db(e[:, tw:].mean()) if T > tw else float('nan'),
                n_nonfinite=nonfinite, curve=e.mean(0).numpy())


def make_plots(out, curves, traj, labels, test, scen):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except Exception:
        print('matplotlib not available; skipping plots')
        return
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    for k, c in curves.items():
        w = max(1, min(20, len(c)))
        cs = np.convolve(c, np.ones(w) / w, mode='valid')
        ax.plot(10 * np.log10(np.maximum(cs, 1e-30)), lw=1.0, label=labels[k])
    ax.set_xlabel('time step')
    ax.set_ylabel('MSE [dB] (moving average)')
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'mse_vs_time.png'), dpi=150)
    if scen.name == 'lorenz' and traj:
        k = len(traj) + 1
        fig = plt.figure(figsize=(2.6 * k, 2.8))
        ax = fig.add_subplot(1, k, 1, projection='3d')
        zt = test.z[0].numpy()
        ax.plot(zt[:, 0], zt[:, 1], zt[:, 2], lw=0.4, color='gray')
        ax.set_title('ground truth', fontsize=7)
        for i, (key, x) in enumerate(traj.items()):
            ax = fig.add_subplot(1, k, i + 2, projection='3d')
            ax.plot(x[:, 0], x[:, 1], x[:, 2], lw=0.4)
            ax.set_title(labels[key], fontsize=6)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'lorenz_trajectories.png'), dpi=150)


def parse_offsets(args):
    if args.offsets:
        return [float(s) for s in args.offsets.split(',')]
    if args.groups == 1:
        return [0.0]
    return np.linspace(-args.offset_span, args.offset_span, args.groups).tolist()


# ---------------------------------------------------------------------------
def cmd_run(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    os.makedirs(args.out, exist_ok=True)
    if args.train_mismatch < 0:
        args.train_mismatch = args.init_mismatch

    arr = SSRArray(parse_offsets(args), args.per_group, args.eta, args.n_quad)
    kstar, zstar, wstar = arr.kappa_opt(1.0)
    print(f'SSR array: M={arr.M} groups x {arr.per_group} comparators (N={arr.N}/feature), '
          f'offsets d={arr.d.tolist()}, eta={args.eta}')
    print(f'  kappa* = {kstar:.4f}, zeta_min(kappa*)/P = {zstar:.4f} '
          f'(1-bit SSR keeps {100 * (1 - zstar):.1f}% of the innovation variance), w* = {wstar.tolist()}')

    gen = torch.Generator().manual_seed(args.seed)
    t0 = time.time()
    scen, train, cv, test = build_scenario(args, arr, gen)
    print(f'scenario {scen.name}: train {train.B}x{train.T}, cv {cv.B}x{cv.T}, test {test.B}x{test.T} '
          f'({time.time() - t0:.1f} s); noise={args.noise}, init_mismatch={args.init_mismatch}, P0={args.P0}')
    if args.dither < 0:
        args.dither = kstar * math.sqrt(float(bdiag(scen.model_filt.R).mean()))

    methods = [s.strip() for s in args.methods.split(',') if s.strip()]
    results, curves, traj, labels = {}, {}, {}, {}
    cal = None
    hdr = f'{"method":<46s}{"MSE[dB]":>9s}{"seq mean+-std":>17s}{"trans[dB]":>10s}{"steady[dB]":>11s}{"nonfin":>7s}{"time[s]":>8s}'
    for name in methods:
        t0 = time.time()
        if name in LEARNED:
            lf = LearnedFilter(scen.model_filt, name, arr=arr, cfg=args,
                               Sig0=torch.diag(train.P0d.mean(0)),
                               in_mult=args.in_mult, out_mult=args.out_mult)
            n_par = sum(p.numel() for p in lf.parameters() if p.requires_grad)
            print(f'training {lf.name}: {n_par} trainable parameters, {args.iters} iterations on {device}')
            train_learned(lf, train, cv, scen.eval_idx, args, device)
            xh, label = run_learned(lf, test, device), lf.name
        elif name in ANALYTIC:
            if name == 'rlsekf' and cal is None:
                cal = calibrate_rlsekf(arr, train.y, train.eta, args.cal_samples)
                print(f'RLS-EKF calibration: sigma={cal["sig"].tolist()}, '
                      f'decoder MSE={cal["Rdec"].tolist()}')
            filt = build_analytic(name, scen, arr, args, cal)
            xh, label = run_analytic(filt, test), filt.name
        else:
            print(f'unknown method "{name}" (choose from {ANALYTIC + LEARNED}); skipped')
            continue
        met = compute_metrics(xh, test.z, scen.eval_idx, args.tr_window)
        curves[name], labels[name] = met.pop('curve'), label
        met.update(label=label, time_s=time.time() - t0)
        results[name] = met
        traj[name] = xh[0].numpy()
        print(hdr)
        for k, r in results.items():
            print(f'{r["label"][:45]:<46s}{r["mse_db"]:9.2f}{r["seq_db_mean"]:9.2f} +-{r["seq_db_std"]:5.2f}'
                  f'{r["transient_db"]:10.2f}{r["steady_db"]:11.2f}{r["n_nonfinite"]:7d}{r["time_s"]:8.1f}')

    with open(os.path.join(args.out, 'results.json'), 'w') as fh:
        json.dump(dict(args=vars(args), kappa_star=kstar, zeta_star=zstar, results=results), fh, indent=2)
    np.savez(os.path.join(args.out, 'curves.npz'), **curves)
    if args.plot:
        make_plots(args.out, curves, traj, labels, test, scen)
    print(f'results written to {os.path.abspath(args.out)}')
    return results


# ---------------------------------------------------------------------------
def cmd_selftest(_args):
    res = []

    def check(name, val, ref, tol):
        ok = abs(val - ref) <= tol
        res.append(ok)
        print(f'[{"PASS" if ok else "FAIL"}] {name}: {val:.5f} (expected {ref:.5f} +- {tol:g})')

    def t(x):
        return torch.tensor([x], dtype=F64)

    print('1) quadrature vs closed form (Gaussian input & noise, M=1, theta=0, N=31)')
    a = SSRArray([0.0], 31, 'gauss')
    for s in (0.3, 1.0, 2.0):
        _, p, C = a.moments(t(1.0), t(s))
        psi = (31.0 * float(C[0, 0, 0]) - 1.0) / 30.0
        check(f'p(sigma={s})', float(p[0, 0]), math.sqrt(2 / math.pi) / math.sqrt(1 + s * s), 3e-4)
        check(f'psi(sigma={s})', psi, (2 / math.pi) * math.asin(1 / (1 + s * s)), 3e-4)
    print('2) Theorem 1 / Table I numbers of the SPL letter')
    for N, ks, zs in ((2, 0.249, 0.327), (31, 1.007, 0.103)):
        k, z, _ = SSRArray([0.0], N, 'gauss').kappa_opt(1.0)
        check(f'sigma_o (M=1, N={N})', k, ks, 0.01)
        check(f'zeta_min (M=1, N={N})', z, zs, 0.002)
    k, z, _ = SSRArray([-0.5, 0.0, 0.5], 21, 'gauss').kappa_opt(1.0)
    check('sigma_o (M=3, N_m=21, theta={0,+-0.5})', k, 1.136, 0.02)
    check('zeta_min (M=3)', z, 0.067, 0.002)
    print('3) generalized Bussgang gain -> sqrt(2/pi)/sqrt(v + kappa^2) (classical as kappa -> 0)')
    a = SSRArray([0.0], 1, 'gauss')
    for v, kk in ((1.0, 0.5), (4.0, 1.0), (1.0, 0.05)):
        mu, p, C = a.moments(t(v), t(kk))
        g, _, _ = a.bussgang(torch.ones(1, 1, dtype=F64), mu, p, C, t(v))
        check(f'gamma (v={v}, kappa={kk})', float(g[0]), math.sqrt(2 / math.pi) / math.sqrt(v + kk * kk), 3e-3)
    print('4) LMMSE weights: gamma = rho and D = rho(1 - rho) v (update = analog KF on decoded innovation)')
    a = SSRArray([-0.5, 0.0, 0.5], 11, 'logistic')
    mu, p, C = a.moments(t(0.7), t(0.9))
    w = a.lmmse_weights(p, C)
    g, _, D = a.bussgang(w, mu, p, C, t(0.7))
    rho = float((w * p).sum(-1)[0]) / 0.7
    check('gamma - rho', float(g[0]) - rho, 0.0, 1e-8)
    check('D - rho(1-rho)v', float(D[0]) - rho * (1 - rho) * 0.7, 0.0, 1e-6)
    print('5) Lorenz Jacobian vs central differences')
    mdl = LorenzModel()
    x = 5.0 * torch.randn(4, 3, dtype=F64) + torch.tensor([0.0, 0.0, 25.0], dtype=F64)
    J = mdl.F(x)
    cols = []
    for i in range(3):
        e = torch.zeros_like(x)
        e[:, i] = 1e-6
        cols.append((mdl.f(x + e) - mdl.f(x - e)) / 2e-6)
    check('max |J - J_fd|', float((J - torch.stack(cols, -1)).abs().max()), 0.0, 1e-5)
    print('6) symmetric alpha-stable sampler')
    gg = torch.Generator().manual_seed(1)
    X = external_noise('sas', (200000,), gg, 2.0, alpha=2.0, scale=1.0, clip=1e12)
    check('alpha=2: Var = 2 c^2', float(X.var()), 2.0, 0.05)
    X = external_noise('sas', (200000,), gg, 2.0, alpha=1.5, scale=1.0, clip=1e12)
    check('alpha=1.5: E cos X = exp(-1)', float(torch.cos(X).mean()), math.exp(-1.0), 0.01)
    print('7) end-to-end smoke test of every method (tiny Lorenz problem)')
    try:
        with tempfile.TemporaryDirectory() as tmp:
            mini = build_parser().parse_args(
                ['run', '--n_train', '16', '--n_cv', '4', '--n_test', '3', '--T_train', '12', '--T_test', '40',
                 '--burn', '50', '--iters', '2', '--cv_every', '1', '--batch', '4', '--cal_samples', '300',
                 '--lag', '3', '--noise', 'sas', '--init_mismatch', '1', '--device', 'cpu', '--out', tmp])
            out = cmd_run(mini)
        ok = len(out) == len(ANALYTIC + LEARNED)
    except Exception:
        traceback.print_exc()
        ok = False
    res.append(ok)
    print(f'[{"PASS" if ok else "FAIL"}] smoke test')
    print(f'\n{sum(res)}/{len(res)} checks passed')


def cmd_prepare_nclt(args):
    trp, trv = prepare_nclt(args.gt_train, args.odom_train, args.hz)
    tep, tev = prepare_nclt(args.gt_test, args.odom_test, args.hz)
    np.savez(args.out, train_pos=trp, train_vel=trv, test_pos=tep, test_vel=tev)
    print(f'wrote {args.out}: train {trp.shape[0]} samples, test {tep.shape[0]} samples at {args.hz} Hz')


# ---------------------------------------------------------------------------
def add_run_args(p):
    a = p.add_argument
    a('--scenario', default='lorenz', choices=['lorenz', 'nclt', 'doppler'])
    a('--methods', default=DEFAULT_METHODS, help='comma-separated method keys (see README)')
    a('--seed', type=int, default=0)
    a('--device', default='auto', help='device for the learned filters (auto/cpu/cuda)')
    a('--threads', type=int, default=0)
    a('--out', default='results')
    a('--plot', action='store_true')
    a('--tr_window', type=int, default=100, help='transient window for the transient/steady split')
    # data
    a('--n_train', type=int, default=1000)
    a('--n_cv', type=int, default=100)
    a('--n_test', type=int, default=50)
    a('--T_train', type=int, default=100)
    a('--T_test', type=int, default=2000)
    a('--burn', type=int, default=500, help='Lorenz burn-in steps to land on the attractor')
    a('--dt', type=float, default=0.02)
    a('--J_data', type=int, default=5)
    a('--J_model', type=int, default=5, help='J=1 reproduces the dynamics-mismatch test')
    a('--q2', type=float, default=1e-3)
    a('--r2', type=float, default=1e-1)
    a('--rot_deg', type=float, default=0.0, help='rotation of the true h (measurement mismatch)')
    # external noise
    a('--noise', default='gauss', choices=['gauss', 'sas', 'mix', 'none'])
    a('--alpha', type=float, default=1.5, help='alpha-stable index')
    a('--noise_scale', type=float, default=0.0, help='SaS scale c (0 -> sqrt(r2/2))')
    a('--mix_eps', type=float, default=0.05)
    a('--mix_kappa', type=float, default=10.0)
    a('--clip', type=float, default=1e4)
    # initialization
    a('--init_mismatch', type=float, default=0.0, help='beta: |x0_hat - x0| in units of state std')
    a('--train_mismatch', type=float, default=-1.0, help='mismatch used in training (-1: same as test)')
    a('--P0', default='informed', choices=['informed', 'small'])
    a('--P0_small', type=float, default=1e-3)
    a('--P0_floor', type=float, default=1e-3)
    # SSR array
    a('--groups', type=int, default=3)
    a('--per_group', type=int, default=11)
    a('--offset_span', type=float, default=0.5, help='offsets d = linspace(-span, span, groups)')
    a('--offsets', default='', help='explicit offsets, e.g. --offsets=-0.5,0,0.5 (note the =)')
    a('--eta', default='gauss', choices=['gauss', 'logistic', 'uniform', 'laplace'])
    a('--n_quad', type=int, default=401)
    # location-scale (aperture) control
    a('--scale_ref', default='hybrid', choices=['model', 'robust', 'hybrid', 'const'])
    a('--kappa_mode', default='opt', choices=['opt', 'ekf', 'fixed'])
    a('--kappa', type=float, default=1.0, help='aperture for kappa_mode=fixed')
    a('--mu_eta', type=float, default=1.0, help='smoothing of the scale reference (proposal mu_eta)')
    a('--mu_s', type=float, default=0.1, help='step of the running median of |e~|')
    a('--mu_sat', type=float, default=0.3, help='extra log-opening per step under full saturation')
    a('--s_const', type=float, default=0.0, help='scale for scale_ref=const (0 -> sqrt(R_jj))')
    a('--inflate', action='store_true', help='SSR-BKF: covariance matching from the robust scale')
    # M-RLS / kappa-EKF feedback (Step 3.4)
    a('--rls_target', default='smoothed', choices=['smoothed', 'posterior', 'oracle'])
    a('--lag', type=int, default=10, help='fixed-lag smoother lag for rls_target=smoothed')
    a('--lam_w', type=float, default=0.1)
    a('--lam_f', type=float, default=0.999)
    a('--rls_P0', type=float, default=10.0)
    a('--huber', type=float, default=1.345)
    a('--rls_init', default='lmmse', choices=['lmmse', 'uniform'])
    a('--Q_kappa', type=float, default=1e-3)
    a('--R_kappa', type=float, default=1.0)
    a('--P_kappa0', type=float, default=2.0)
    a('--kappa_min', type=float, default=0.05)
    a('--kappa_max', type=float, default=10.0)
    # baselines
    a('--dither', type=float, default=-1.0, help='fixed dither of bkf_dither (-1 -> kappa* sqrt(R))')
    a('--cal_samples', type=int, default=20000, help='analog samples for RLS-EKF calibration')
    # learned filters
    a('--iters', type=int, default=1000)
    a('--batch', type=int, default=32)
    a('--lr', type=float, default=1e-3)
    a('--wd', type=float, default=1e-4)
    a('--in_mult', type=int, default=5)
    a('--out_mult', type=int, default=40)
    a('--cv_every', type=int, default=50)
    a('--clip_grad', type=float, default=10.0)
    a('--st_grad', action='store_true', help='SSR-BKNet: Bussgang straight-through gradient')
    a('--learn_w', action='store_true', help='SSR-BKNet: learn the spatial weights end-to-end')
    # NCLT
    a('--nclt_npz', default='')
    a('--nclt_hz', type=float, default=1.0)
    # Doppler surrogate
    a('--dop_dt', type=float, default=0.01)
    a('--dop_amp', type=float, default=1.0)
    a('--dop_qphi', type=float, default=1e-4)
    a('--dop_qf', type=float, default=1e-6)
    a('--dop_qfd', type=float, default=1e-4)
    a('--dop_r2', type=float, default=1.0)
    a('--dop_frange', type=float, default=5.0)
    a('--dop_fdstd', type=float, default=0.1)


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest='cmd', required=True)
    add_run_args(sub.add_parser('run', help='generate data, train/run all methods, report MSE'))
    sub.add_parser('selftest', help='numerical checks + tiny end-to-end smoke test')
    pn = sub.add_parser('prepare-nclt', help='build the NCLT .npz from raw CSV files')
    pn.add_argument('--gt_train', required=True)
    pn.add_argument('--odom_train', required=True)
    pn.add_argument('--gt_test', required=True)
    pn.add_argument('--odom_test', required=True)
    pn.add_argument('--hz', type=float, default=1.0)
    pn.add_argument('--out', default='nclt.npz')
    return ap


def main():
    args = build_parser().parse_args()
    if args.cmd == 'run':
        cmd_run(args)
    elif args.cmd == 'selftest':
        cmd_selftest(args)
    else:
        cmd_prepare_nclt(args)


if __name__ == '__main__':
    main()
