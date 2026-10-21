#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Redraw v3 curve plots from existing local CSVs (analysis/curves).

Addresses 6 review comments; all outputs carry the _v3 suffix and never
overwrite existing files.  No training, no server access.

Outputs (analysis/curves/):
  <TAG>_loss_curves_v3.png            x4  loss vs training progress
  LConf_DEO_scatter_v3.png            L_Conf vs same-round DEO (DUTCH / COMPAS D01)
  L_PA_cross_dataset_v3.png           L_PA rolling median, 4 datasets
  COMPAS_Uniform_K20_baselines_convergence_v3.png   PDFFed vs FedAvg vs FairFed
  summary_trends_v3.png               2x3 cross-dataset summary
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BASE = os.path.dirname(os.path.abspath(__file__))
CURVES = os.path.join(BASE, 'curves')
RAW = os.path.join(CURVES, 'raw')

TAGS = [
    ('COMPAS_Dirichlet01_PDFFed_K20', 'COMPAS (Dirichlet 0.1)', 'tab:blue'),
    ('COMPAS_Uniform_PDFFed_K20',     'COMPAS (Uniform)',       'tab:orange'),
    ('DUTCH_Dirichlet01_PDFFed_K20',  'Dutch (Dirichlet 0.1)',  'tab:green'),
    ('CREMA-D_Speaker_PDFFed_K20',    'CREMA-D (Speaker)',      'tab:red'),
]
LOSS_COLS = ['origin_task_loss', 'L_PA', 'L_Conf', 'L_l2l',
             'L_l2g', 'L_g2l', 'cov_abs_diag']
COLLAPSE_COLS = ['L_l2l', 'L_l2g', 'L_g2l', 'cov_abs_diag', 'L_Conf']


def fp(tag, kind):
    return os.path.join(CURVES, '%s_%s.csv' % (tag, kind))


def load_loss(tag):
    return pd.read_csv(fp(tag, 'loss'))


def load_global(tag):
    return pd.read_csv(fp(tag, 'global'))


def per_round_mean(G, col):
    g = G.dropna(subset=[col]).groupby('communication_round')[col].mean()
    return g.index.values.astype(float), g.values.astype(float)


def rolling(y, w, how='mean', mp=3):
    s = pd.Series(np.asarray(y, dtype=float))
    r = s.rolling(w, min_periods=mp).median() if how == 'median' \
        else s.rolling(w, min_periods=mp).mean()
    return r.values


def collapse_point(med, thresh=1e-2, tail_frac=0.10):
    """First progress at which the rolling median enters and stays < thresh.

    Returns None when the series never exceeds thresh or never settles
    below it for the final `tail_frac` of training.
    """
    v = np.asarray(med, dtype=float)
    v = v[~np.isnan(v)]
    above = np.where(v >= thresh)[0]
    if len(above) == 0:
        return None
    last = above[-1]
    if last < (1.0 - tail_frac) * (len(v) - 1):
        return (last + 1) / (len(v) - 1)
    return None


# ----------------------------------------------------------------------
# 1. loss curves v3: x = training progress (loss-block index / total)
# ----------------------------------------------------------------------
def plot_loss_v3(tag):
    L = load_loss(tag).sort_values('step', kind='stable').reset_index(drop=True)
    x = L['step'].values.astype(float) / max(L['step'].max(), 1)
    N = len(L)
    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    finite_pos, notes = [], []
    for ci, col in enumerate(LOSS_COLS):
        c = colors[ci % len(colors)]
        s = L[col].astype(float)
        med = rolling(s, 200, 'median', mp=50)
        qlo = s.rolling(200, min_periods=50).quantile(0.10).values
        qhi = s.rolling(200, min_periods=50).quantile(0.90).values
        finite_pos += s[s > 0].tolist()
        ax.fill_between(x, qlo, qhi, color=c, alpha=0.15, linewidth=0)
        ax.plot(x, med, color=c, lw=1.7, label=col)
        if col in COLLAPSE_COLS:
            p = collapse_point(med)
            if p is not None:
                notes.append((p, '%s collapses to ~0 by progress~%.2f' % (col, p), c))
    # staggered annotation text at the top of the axes
    for k, (p, txt, c) in enumerate(notes):
        ax.text(p, 0.97 - 0.05 * k, txt, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8.5, color=c,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', ec=c,
                          alpha=0.85, lw=0.6))
    if finite_pos:
        try:
            ax.set_yscale('symlog', linthresh=max(min(finite_pos) * 0.5, 1e-6))
        except Exception:
            pass
    ax.set_xlim(0, 1)
    ax.set_xlabel('Training progress (loss-block index / total)')
    ax.set_ylabel('Loss value (symlog)')
    ax.set_title('%s: training loss terms - pooled 200-block rolling median '
                 'with 10-90%% band' % tag, fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    out = os.path.join(CURVES, '%s_loss_curves_v3.png' % tag)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out, '| collapse annotations:', len(notes))
    return notes


# ----------------------------------------------------------------------
# 2. L_Conf vs same-round DEO scatter (Theorem 6 empirical)
# ----------------------------------------------------------------------
def conf_deo_points(tag, bucket=50):
    """50-block-bucket mean L_Conf paired with nearest-round DEO (one repeat)."""
    L = load_loss(tag)
    rep = L['repeat'].value_counts().idxmax()          # repeat with most blocks
    Lr = L[L['repeat'] == rep].sort_values('step', kind='stable')
    G = load_global(tag)
    Gr = G[G['repeat'] == rep].dropna(subset=['DEO'])
    Gd = Gr.groupby('communication_round')['DEO'].mean().sort_index()
    rounds = Gd.index.values.astype(float)
    b = (Lr['step'].values // bucket)
    g = Lr.groupby(b)
    lconf = g['L_Conf'].mean().values
    rmed = g['round'].median().values
    nb = len(lconf)
    pos = np.clip(np.searchsorted(rounds, rmed), 1, len(rounds) - 1)
    left, right = rounds[pos - 1], rounds[pos]
    nearest = np.where(np.abs(rmed - left) <= np.abs(rmed - right), left, right)
    deo = Gd.reindex(nearest).values
    ok = ~(np.isnan(deo) | np.isnan(lconf))
    deo, lconf = deo[ok], lconf[ok]
    r = float(np.corrcoef(deo, lconf)[0, 1])
    return deo, lconf, r, nb, int(rep)


def plot_conf_deo_scatter(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, (tag, label, res) in zip(axes, results):
        deo, lconf, r, nb, rep = res
        prog = np.arange(len(deo)) / max(len(deo) - 1, 1)
        sc = ax.scatter(deo, lconf, c=prog, cmap='viridis', s=20, alpha=0.85,
                        linewidths=0.3, edgecolors='k')
        cb = fig.colorbar(sc, ax=ax, pad=0.01)
        cb.set_label('Training progress (bucket / total)', fontsize=9)
        ax.set_xlabel('DEO (same communication round, nearest-round aligned)')
        ax.set_ylabel('L_Conf (50-block bucket mean)')
        ax.set_title('%s\nPearson r = %.3f  (n=%d buckets, repeat %d)'
                     % (label, r, len(deo), rep), fontsize=10.5)
        ax.grid(True, alpha=0.3)
    fig.suptitle('L_Conf vs DEO during training - empirical check of Theorem 6',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = os.path.join(CURVES, 'LConf_DEO_scatter_v3.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


# ----------------------------------------------------------------------
# 3. L_PA across datasets
# ----------------------------------------------------------------------
def lpa_curve(tag):
    L = load_loss(tag)
    g = L.dropna(subset=['L_PA']).groupby('round')['L_PA'].mean().sort_index()
    rounds = g.index.values.astype(float)
    total = rounds.max()
    med = rolling(g.values, 25, 'median', mp=5)
    return rounds / total, med


def plot_lpa_cross(ax=None, legend=True):
    solo = ax is None
    if solo:
        fig, ax = plt.subplots(figsize=(8.5, 5.2))
    for tag, label, color in TAGS:
        x, y = lpa_curve(tag)
        ax.plot(x, y, color=color, lw=1.8, label='%s (L_PA)' % label)
    ax.set_xlabel('Training progress (round / total rounds)')
    ax.set_ylabel('L_PA')
    ax.set_title('L_PA contraction across datasets (Theorem 2), '
                 '25-round rolling median', fontsize=11)
    ax.set_xlim(0, 1)
    ax.grid(True, alpha=0.3)
    if legend:
        ax.legend(fontsize=9)
    if solo:
        fig.tight_layout()
        out = os.path.join(CURVES, 'L_PA_cross_dataset_v3.png')
        fig.savefig(out, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print('wrote', out)


# ----------------------------------------------------------------------
# 4. baseline overlay: PDFFed vs FedAvg vs FairFed (COMPAS Uniform, K=20)
# ----------------------------------------------------------------------
def load_baselines():
    out = {}
    for alg in ['FedAvg', 'FairFed']:
        p = os.path.join(RAW, 'COMPAS_Uniform_%s_K20_global.csv' % alg)
        if os.path.exists(p):
            out[alg] = pd.read_csv(p)
    return out


def plot_baselines():
    G = load_global('COMPAS_Uniform_PDFFed_K20')
    bl = load_baselines()
    series = {}
    r_pd, d_pd = per_round_mean(G, 'DEO')
    _, h_pd = per_round_mean(G, 'HM')
    series['PDFFed'] = (r_pd, rolling(d_pd, 15), rolling(h_pd, 15))
    for alg, B in bl.items():
        r, d = per_round_mean(B, 'DEO')
        _, h = per_round_mean(B, 'HM')
        series[alg] = (r, rolling(d, 15), rolling(h, 15))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    axr = ax.twinx()
    styles = {'PDFFed': ('tab:blue', '-'),
              'FedAvg': ('tab:orange', '--'),
              'FairFed': ('tab:green', '-.')}
    drift_txt = []
    for name, (r, d, h) in series.items():
        c, ls = styles[name]
        ax.plot(r, d, color=c, ls=ls, lw=1.8, label='%s - DEO (left)' % name)
        axr.plot(r, h, color=c, ls=ls, lw=1.4, alpha=0.65,
                 label='%s - HM (right)' % name)
        d0, d1 = np.nanmean(d[:15]), np.nanmean(d[-15:])
        drift_txt.append('%s: %.3f -> %.3f (drift %+.3f)'
                         % (name, d0, d1, d1 - d0))
        # end-value marker for DEO
        ax.annotate('%.3f' % d[-1], xy=(r[-1], d[-1]), xytext=(4, 0),
                    textcoords='offset points', fontsize=8.5, color=c, va='center')
    ax.set_xlabel('Communication round')
    ax.set_ylabel('DEO (15-eval rolling mean)')
    axr.set_ylabel('HM (15-eval rolling mean)', alpha=0.8)
    axr.tick_params(axis='y', colors='0.45')
    ax.set_xlim(0, 500)
    ax.set_title('COMPAS Uniform K=20: PDFFed vs FedAvg vs FairFed - '
                 'DEO (solid) and HM (translucent dashed)', fontsize=11)
    ax.grid(True, alpha=0.3)
    l1, lb1 = ax.get_legend_handles_labels()
    l2, lb2 = axr.get_legend_handles_labels()
    ax.legend(l1 + l2, lb1 + lb2, fontsize=8, ncol=2, loc='upper left')
    note = 'DEO first-15 -> last-15 eval mean:\n' + '\n'.join(drift_txt)
    if 'FairFed' in bl and 'FedAvg' in bl:
        fa, ff = bl['FedAvg'], bl['FairFed']
        same = (len(fa) == len(ff) and
                np.allclose(fa['DEO'].values, ff['DEO'].values) and
                np.allclose(fa['HM'].values, ff['HM'].values))
        if same:
            note += ('\nNote: FedAvg and FairFed logged identical metric '
                     'sequences (147/147 evals); their curves overlap.')
    ax.text(0.985, 0.30, note, transform=ax.transAxes, ha='right', va='top',
            fontsize=8, bbox=dict(boxstyle='round,pad=0.35', fc='white',
                                  ec='0.6', alpha=0.9))
    fig.tight_layout()
    out = os.path.join(CURVES, 'COMPAS_Uniform_K20_baselines_convergence_v3.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)
    print('baseline drift summary:')
    for line in drift_txt:
        print('  ', line)
    return series


# ----------------------------------------------------------------------
# 5. summary trends v3 (2x3)
# ----------------------------------------------------------------------
def plot_summary(scatter_res, bl_series):
    fig, axes = plt.subplots(2, 3, figsize=(19, 10.5))
    (axa, axb, axc), (axd, axe, axf) = axes

    # (a) HM / (b) DEO convergence, PDFFed 4 datasets (+ baseline overlay)
    for ax, col in ((axa, 'HM'), (axb, 'DEO')):
        for tag, label, color in TAGS:
            G = load_global(tag)
            r, y = per_round_mean(G, col)
            ax.plot(r / r.max(), rolling(y, 15), color=color, lw=1.8,
                    label=label)
        for name, ls in (('FedAvg', '--'), ('FairFed', '-.')):
            if name in bl_series:
                r, d, h = bl_series[name]
                y = h if col == 'HM' else d
                ax.plot(r / 500.0, y, color='0.35', ls=ls, lw=1.4,
                        label='COMPAS (U.) %s' % name)
        ax.set_xlabel('Training progress (round / total rounds)')
        ax.set_ylabel(col)
        ax.set_title('(%s) %s convergence (15-eval rolling mean)'
                     % ('a' if col == 'HM' else 'b', col), fontsize=11)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=7.5, ncol=2)

    # (c) L_PA across datasets
    plot_lpa_cross(ax=axc, legend=True)
    axc.set_title('(c) ' + axc.get_title(), fontsize=11)

    # (d) CREMA-D ACC vs constant predictor
    G = load_global('CREMA-D_Speaker_PDFFed_K20')
    r, acc = per_round_mean(G, 'ACC')
    acc_s = rolling(acc, 15)
    cp = float(np.mean(acc[:5]))            # constant-predictor accuracy
    axd.plot(r, acc_s, color='tab:red', lw=1.8, label='ACC (15-eval rolling mean)')
    axd.axhline(cp, color='0.3', ls='--', lw=1.2,
                label='Constant predictor = negative-class share (%.3f)' % cp)
    end = 0
    for i, v in enumerate(acc):
        if v >= cp - 0.005:
            end = r[i]
        else:
            break
    if end > 0:
        axd.axvspan(0, end, color='0.5', alpha=0.18)
        axd.text(end * 0.55, axd.get_ylim()[0] + 0.02,
                 'constant-predictor phase\n(rounds 1-%d)' % end,
                 fontsize=8.5, ha='center')
    axd.set_xlabel('Communication round')
    axd.set_ylabel('ACC')
    axd.set_title('(d) CREMA-D ACC vs constant predictor', fontsize=11)
    axd.grid(True, alpha=0.3)
    axd.legend(fontsize=8.5, loc='lower left')

    # (e) L_Conf vs DEO scatter conclusion
    for (tag, label, res), color in zip(scatter_res,
                                        ('tab:green', 'tab:blue')):
        deo, lconf, rr, nb, rep = res
        axe.scatter(deo, lconf, s=14, alpha=0.6, color=color,
                    label='%s (r=%.2f)' % (label, rr))
    rs = [res[2][2] for res in scatter_res]
    if all(v >= 0.3 for v in rs):
        concl = ('L_Conf co-moves with DEO during training\n'
                 '(Pearson r: %.2f / %.2f) - empirical\nsupport for Theorem 6.'
                 % tuple(rs))
    else:
        concl = ('L_Conf-DEO correlation is weak at 50-block\n'
                 'granularity (Pearson r: %.2f / %.2f);\n'
                 'Theorem 6 not clearly supported at this\ngranularity.'
                 % tuple(rs))
    axe.text(0.02, 0.98, concl, transform=axe.transAxes, va='top', fontsize=9.5,
             bbox=dict(boxstyle='round,pad=0.4', fc='lemonchiffon',
                       ec='0.5', alpha=0.95))
    axe.set_xlabel('DEO (same round)')
    axe.set_ylabel('L_Conf (50-block bucket mean)')
    axe.set_title('(e) L_Conf vs DEO (Theorem 6)', fontsize=11)
    axe.grid(True, alpha=0.3)
    axe.legend(fontsize=8.5, loc='lower right')

    axf.axis('off')
    fig.suptitle('PDFFed K20: cross-dataset summary (v3)', fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(CURVES, 'summary_trends_v3.png')
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print('wrote', out)


def main():
    all_notes = {}
    for tag, _, _ in TAGS:
        all_notes[tag] = plot_loss_v3(tag)

    scatter_res = []
    for tag, label, _ in TAGS:
        if tag in ('DUTCH_Dirichlet01_PDFFed_K20',
                   'COMPAS_Dirichlet01_PDFFed_K20'):
            scatter_res.append((tag, label, conf_deo_points(tag)))
    plot_conf_deo_scatter(scatter_res)
    for tag, label, (deo, lconf, r, nb, rep) in scatter_res:
        print('L_Conf-DEO %s: Pearson r=%.4f, n=%d buckets (repeat %d)'
              % (label, r, nb, rep))

    plot_lpa_cross()
    bl_series = plot_baselines()
    plot_summary(scatter_res, bl_series)

    print('\ncollapse detection (threshold 1e-2, on 200-block rolling median):')
    for tag, notes in all_notes.items():
        for p, txt, _ in notes:
            print('  %s: %s' % (tag, txt))
        if not notes:
            print('  %s: no column collapsed below 1e-2' % tag)


if __name__ == '__main__':
    main()
