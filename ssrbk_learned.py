# -*- coding: utf-8 -*-
"""Learned filters: KalmanNet (analog), BKNet (Jung et al.) and the proposed SSR-BKNet.

All three share GainNet = KalmanNet architecture #2 (GRU_Q, GRU_Sigma, GRU_S + FC1..FC7).
With in_mult=5, out_mult=40 it has exactly 23,928 parameters for m = n = 3 (Lorenz) and
107,656 for the NCLT model, i.e. the BKNet sizes reported by Jung et al.
SSR-BKNet adds one FC branch (FC8) for the SSR side information.
"""
import copy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as Fnn

from ssrbk_common import bmv, bdiag, qsign


class GainNet(nn.Module):
    def __init__(self, m, n, Q0, Sig0, S0, in_mult=5, out_mult=40, n_extra=0):
        super().__init__()
        self.m, self.n, self.n_extra = m, n, n_extra
        self.FC5 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.GRU_Q = nn.GRU(m * in_mult, m * m)
        self.FC6 = nn.Sequential(nn.Linear(m, m * in_mult), nn.ReLU())
        self.GRU_Sigma = nn.GRU(m * m + m * in_mult, m * m)
        self.FC1 = nn.Sequential(nn.Linear(m * m, n * n), nn.ReLU())
        self.FC7 = nn.Sequential(nn.Linear(2 * n, 2 * n * in_mult), nn.ReLU())
        d_extra = 0
        if n_extra > 0:
            self.FC8 = nn.Sequential(nn.Linear(n_extra, n_extra * in_mult), nn.ReLU())
            d_extra = n_extra * in_mult
        self.GRU_S = nn.GRU(n * n + 2 * n * in_mult + d_extra, n * n)
        d2 = n * n + m * m
        self.FC2 = nn.Sequential(nn.Linear(d2, d2 * out_mult), nn.ReLU(),
                                 nn.Linear(d2 * out_mult, n * m))
        self.FC3 = nn.Sequential(nn.Linear(n * n + n * m, m * m), nn.ReLU())
        self.FC4 = nn.Sequential(nn.Linear(2 * m * m, m * m), nn.ReLU())
        self.register_buffer('prior_Q', Q0.reshape(-1).float())
        self.register_buffer('prior_Sigma', Sig0.reshape(-1).float())
        self.register_buffer('prior_S', S0.reshape(-1).float())

    def init_hidden(self, B):
        self.h_Q = self.prior_Q.view(1, 1, -1).expand(1, B, -1).contiguous()
        self.h_Sigma = self.prior_Sigma.view(1, 1, -1).expand(1, B, -1).contiguous()
        self.h_S = self.prior_S.view(1, 1, -1).expand(1, B, -1).contiguous()

    def forward(self, obs_diff, innov, evol_diff, upd_diff, extra=None):
        def nz(x):
            return Fnn.normalize(x, p=2, dim=-1, eps=1e-12).unsqueeze(0)
        obs_diff, innov, evol_diff, upd_diff = nz(obs_diff), nz(innov), nz(evol_diff), nz(upd_diff)
        out_Q, self.h_Q = self.GRU_Q(self.FC5(evol_diff), self.h_Q)
        out_Sig, self.h_Sigma = self.GRU_Sigma(torch.cat([out_Q, self.FC6(upd_diff)], -1), self.h_Sigma)
        parts = [self.FC1(out_Sig), self.FC7(torch.cat([obs_diff, innov], -1))]
        if self.n_extra > 0:
            parts.append(self.FC8(extra.unsqueeze(0)))
        out_S, self.h_S = self.GRU_S(torch.cat(parts, -1), self.h_S)
        out_FC2 = self.FC2(torch.cat([out_Sig, out_S], -1))
        out_FC3 = self.FC3(torch.cat([out_S, out_FC2], -1))
        self.h_Sigma = self.FC4(torch.cat([out_Sig, out_FC3], -1))
        return out_FC2.squeeze(0).reshape(-1, self.m, self.n)


class LearnedFilter(nn.Module):
    """kind = 'knet'     : KalmanNet on analog y (innovation y - h(z_prior))
              'bknet'    : BKNet, 1 comparator / feature, threshold h(z_prior), update K r (Eq. 53)
              'ssrbknet' : SSR array with location-scale control (robust scale reference, kappa*),
                           LMMSE spatial decoding -> unbiased residual e~ (Step 3.2), learned gain
                           on e~ plus SSR side information [log aperture, saturation, e~/scale]."""

    NAMES = {'knet': 'KalmanNet (analog, full precision)', 'bknet': 'BKNet (Jung et al.)',
             'ssrbknet': 'SSR-BKNet (proposed)'}

    def __init__(self, model, kind, arr=None, cfg=None, Sig0=None, in_mult=5, out_mult=40):
        super().__init__()
        self.model, self.kind, self.arr = model, kind, arr
        self.name = self.NAMES[kind]
        m, n = model.m, model.n
        S0 = model.R if kind == 'knet' else torch.eye(n, dtype=torch.float64)
        self.net = GainNet(m, n, model.Q, Sig0, S0, in_mult, out_mult,
                           n_extra=3 * n if kind == 'ssrbknet' else 0)
        if kind == 'ssrbknet':
            self.mu_s, self.mu_sat = float(cfg.mu_s), float(cfg.mu_sat)
            k, _, w = arr.kappa_opt(1.0)
            self.kstar = k
            one = torch.tensor([1.0], dtype=torch.float64)
            mu, p, _ = arr.moments(one, torch.tensor([k], dtype=torch.float64))
            self.register_buffer('mu_vec', mu[0].float())
            self.register_buffer('p_vec', p[0].float())
            if cfg.learn_w:
                self.w = nn.Parameter(w.float().clone())
            else:
                self.register_buffer('w', w.float().clone())

    def forward(self, y, eta, z0, P0d, st_grad=False):
        mdl, kind = self.model, self.kind
        B, T, _ = y.shape
        self.net.init_hidden(B)
        z_post = z_post_prev = z_prior_prev = z0
        with torch.no_grad():
            h0 = mdl.h(z0)
        prev = torch.zeros_like(h0) if kind == 'bknet' else h0
        if kind == 'ssrbknet':
            gam = (self.w * self.p_vec).sum()          # Bussgang gain of w^T s for u ~ N(0, 1)
            mean = (self.w * self.mu_vec).sum()
            with torch.no_grad():
                H0 = mdl.H(z0)
                P0 = H0 @ torch.diag_embed(P0d) @ H0.transpose(-1, -2) + mdl.Rm(z0)
                scale0 = bdiag(P0).clamp_min(1e-12).sqrt()
                m_rob = 0.6745 * scale0
                kap = torch.full_like(scale0, self.kstar)
        outs = []
        for t in range(T):
            z_prior = mdl.f(z_post)
            y_pred = mdl.h(z_prior)
            extra = None
            if kind == 'knet':
                obs = y[:, t]
                upd = obs - y_pred
                obs_diff, innov, prev = obs - prev, upd, obs
            elif kind == 'bknet':
                with torch.no_grad():
                    r = qsign(y[:, t] - y_pred)          # comparator is not differentiated
                upd = r
                obs_diff, innov, prev = r - prev, r, r
            else:
                with torch.no_grad():
                    scale = (m_rob / 0.6745).clamp_min(1e-6)
                    sg = self.arr.bits(y[:, t], y_pred, scale, kap, eta[:, t])
                upd = scale * ((self.w * sg).sum(-1) - mean) / gam
                if st_grad:                              # Bussgang straight-through: d e~/d yhat = -1
                    upd = upd + (y_pred.detach() - y_pred)
                xt = y_pred.detach() + upd               # reconstructed analog observation x~_t
                obs_diff, innov = xt - prev, upd
                prev = xt.detach()
                with torch.no_grad():
                    sat_g = (sg.abs() >= 1.0 - 1e-6).to(sg.dtype)
                    extra = torch.cat([torch.log(scale / scale0), sat_g.mean(-1),
                                       (upd / scale).clamp(-5.0, 5.0)], -1)
                    ind = (upd.abs() > m_rob).to(sg.dtype)
                    sat_all = (sg.mean(-1).abs() >= 1.0 - 1e-6).to(sg.dtype)
                    m_rob = (m_rob * torch.exp(self.mu_s * (ind - 0.5) + self.mu_sat * sat_all)
                             ).clamp(1e-6, 1e6)
            K = self.net(obs_diff, innov, z_post - z_post_prev, z_post - z_prior_prev, extra)
            z_new = z_prior + bmv(K, upd)
            z_post_prev, z_prior_prev, z_post = z_post, z_prior, z_new
            outs.append(z_post)
        return torch.stack(outs, 1)


def seq_loss(xh, z, idx):
    return ((xh[..., idx] - z[..., idx]) ** 2).sum(-1).mean()


@torch.no_grad()
def eval_loss(lf, data, idx, chunk=128):
    lf.eval()
    tot, cnt = 0.0, 0
    for i in range(0, data.B, chunk):
        b = data.slice(slice(i, i + chunk))
        e = ((lf(b.y, b.eta, b.z0, b.P0d)[..., idx] - b.z[..., idx]) ** 2).sum(-1)
        e = torch.nan_to_num(e, nan=1e12, posinf=1e12, neginf=1e12)
        tot, cnt = tot + float(e.sum()), cnt + e.numel()
    return tot / max(cnt, 1)


def train_learned(lf, train, cv, idx, args, device, log=print):
    """Supervised BPTT training (Jung et al., Eqs. 54-58), best-CV checkpoint is kept."""
    lf.to(device)
    tr, cvd = train.to(device, torch.float32), cv.to(device, torch.float32)
    opt = torch.optim.Adam(lf.parameters(), lr=args.lr, weight_decay=args.wd)
    best, best_state = float('inf'), copy.deepcopy(lf.state_dict())
    t0 = time.time()
    for it in range(1, args.iters + 1):
        lf.train()
        bi = torch.randint(0, tr.B, (min(args.batch, tr.B),)).to(device)
        b = tr.slice(bi)
        loss = seq_loss(lf(b.y, b.eta, b.z0, b.P0d, st_grad=args.st_grad), b.z, idx)
        opt.zero_grad(set_to_none=True)
        if torch.isfinite(loss):
            loss.backward()
            nn.utils.clip_grad_norm_(lf.parameters(), args.clip_grad)
            opt.step()
        if it % args.cv_every == 0 or it == args.iters:
            cvl = eval_loss(lf, cvd, idx)
            if cvl < best:
                best, best_state = cvl, copy.deepcopy(lf.state_dict())
            log(f'    [{lf.name}] it {it:5d}  train {10 * math.log10(max(float(loss), 1e-30)):7.2f} dB'
                f'  cv {10 * math.log10(max(cvl, 1e-30)):7.2f} dB  ({time.time() - t0:.0f} s)')
    lf.load_state_dict(best_state)
    return best


@torch.no_grad()
def run_learned(lf, data, device, chunk=64):
    lf.eval()
    lf.to(device)
    outs = []
    for i in range(0, data.B, chunk):
        b = data.slice(slice(i, i + chunk)).to(device, torch.float32)
        outs.append(lf(b.y, b.eta, b.z0, b.P0d).double().cpu())
    return torch.cat(outs, 0)
