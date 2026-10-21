#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Extract per-step loss curves and per-round global convergence metrics from
federated training text logs (streaming line-by-line, multi-GB safe).

Usage:
  python extract_curves.py --log <1.txt> --tag <TAG> [--outdir DIR]
                           [--multi-stream] [--start-line N] [--no-plot]

Outputs under --outdir:
  <TAG>_loss.csv        one row per loss block (= one training step)
  <TAG>_global.csv      one row per communication-round global test
  <TAG>_loss_curves.png loss terms (vs step for single-stream, vs round for
                        multi-stream aggregate)
  <TAG>_convergence.png ACC/HM and DEO/SPD vs communication round

Log format handled:
  * loss block = ONE log record spanning multiple PHYSICAL lines:
        ...INFO    : ### Origin task loss: <v> ;
        proto_alignment_loss (L_PA): <v> ;
        delta_conf_loss (L_Conf): <v> ; ...
        in Batch_id:0 of Epoch:0 in Client:0. ###
    All 'name: value ;' items are captured generically ('name (ALIAS)' uses
    the alias, e.g. L_PA; 'cov_abs(diag)' -> cov_abs_diag).
  * global test row:  'ACC: v, DEO: v, SPD:v, FR: v, HM: v, ACC_gap: v'
    preceded by an optional 'per-group accuracy: {k: v, ...}' line and
    followed a few lines later by 'Global Model testing at Communication R/ T'
    which supplies the communication round.
  * rows are grouped by invocation ('exp_repeat_times' echo) and repeat
    (communication-round counter wrap on 'Select clients' lines, validated
    against 'repeat_K' checkpoint paths).

--multi-stream: for log files where several concurrent processes interleaved
  their lines (e.g. COMPAS Uniform on gpu30). Repeat detection is disabled
  (all rows repeat=0); each global metric row is attributed to the round of
  the NEXT 'Global Model testing' marker within a 12-line window (this also
  filters periodic extra tests that carry no such marker); plots show the
  per-round mean with a min-max band over all passes.
"""
import argparse
import csv
import os
import re
from collections import OrderedDict, defaultdict

NUM = r'[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?'
ROUND_START_RE = re.compile(r'Communication Round:\s*(\d+);\s*Select clients')
PERCLIENT_RE = re.compile(
    r'Communication Round:\s*(\d+)\s*/\s*\d+;\s*Client:\s*(\d+)\s*/\s*\d+')
COST_RE = re.compile(r'Communication Round (\d+)\s+Communication Cost')
GMT_RE = re.compile(r'Global Model testing at Communication (\d+)/\s*(\d+)')
REPEAT_HDR_RE = re.compile(r'exp_repeat_times\s*:')
REPEAT_ID_RE = re.compile(r'repeat_(\d+)')
PAIR_RE = re.compile(r'([A-Za-z_]\w*)\s*[:]\s*(%s)' % NUM)
ITEM_RE = re.compile(r'([A-Za-z][\w ()/]*?)\s*[:\uff1a]\s*(%s)\s*;' % NUM)
TERM_RE = re.compile(
    r'in Batch_id:\s*(\d+)\s+of Epoch:\s*(\d+)\s+in Client:\s*(\d+)')
GROUP_ACC_RE = re.compile(r'per-group accuracy:\s*\{([^}]*)\}')
GPAIR_RE = re.compile(r'([^:,{}]+?)\s*:\s*(%s)' % NUM)
TS_PREFIX_RE = re.compile(
    r'^\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}[,.]\d+\s+\w+\s*:\s*')
GMT_WINDOW = 12          # lines between a metric row and its GMT marker


def norm_loss_name(raw):
    """'proto_alignment_loss (L_PA)' -> 'L_PA'; 'cov_abs(diag)' -> 'cov_abs_diag'."""
    raw = raw.strip()
    m = re.match(r'^(.*?\S)\s+\(([^()]+)\)$', raw)
    if m:
        alias = m.group(2).strip()
        if alias:
            return alias
    return re.sub(r'[^A-Za-z0-9]+', '_', raw).strip('_').lower()


def parse_log(path, multi_stream=False, start_line=1):
    inv = -1
    rep = 0
    cur_round = 0
    last_round = 0
    step = 0
    last_groups = []
    cost_round = None
    loss_rows = []
    glob_rows = []
    metric_cand = []        # metric rows awaiting their GMT marker
    loss_cols = []
    glob_cols = []
    n_inv = defaultdict(lambda: {'loss': 0, 'glob': 0})
    stats = {'gmt': 0, 'metric_seen': 0, 'metric_unpaired': 0}
    block_buf = None
    line_idx = 0

    def close_block(buf):
        nonlocal step
        text = '\n'.join(buf)
        text = TS_PREFIX_RE.sub('', text, count=1)
        items = OrderedDict()
        for im in ITEM_RE.finditer(text):
            items[norm_loss_name(im.group(1))] = float(im.group(2))
        tm = TERM_RE.search(text)
        row = {'inv': inv, 'rep': rep, 'round': cur_round,
               'client': int(tm.group(3)) if tm else -1,
               'epoch': int(tm.group(2)) if tm else -1,
               'batch': int(tm.group(1)) if tm else -1,
               'step': step}
        row.update(items)
        for k in items:
            if k not in loss_cols:
                loss_cols.append(k)
        loss_rows.append(row)
        n_inv[inv]['loss'] += 1
        step += 1

    with open(path, 'r', encoding='utf-8', errors='replace') as f:
        for line in f:
            line_idx += 1
            if line_idx < start_line:
                continue

            if REPEAT_HDR_RE.search(line):
                if block_buf is not None:       # malformed cut block
                    close_block(block_buf)
                    block_buf = None
                inv += 1
                rep = 0
                cur_round = 0
                last_round = 0
                step = 0
                cost_round = None
                metric_cand = []
                continue

            # ---- multi-physical-line loss block buffering ----
            if block_buf is not None:
                block_buf.append(line)
                if '###' in line:
                    close_block(block_buf)
                    block_buf = None
                continue
            if 'Origin task loss' in line:
                block_buf = [line]
                if line.rstrip().endswith('###'):   # defensive: single-line block
                    close_block(block_buf)
                    block_buf = None
                continue

            # ---- global test marker: assign pending metric rows ----
            m = GMT_RE.search(line)
            if m:
                stats['gmt'] += 1
                rnd = int(m.group(1))
                for cand in metric_cand:
                    if line_idx - cand['line'] <= GMT_WINDOW:
                        cand['round'] = rnd
                        glob_rows.append(cand)
                        n_inv[cand['inv']]['glob'] += 1
                    else:
                        stats['metric_unpaired'] += 1
                metric_cand = []
                continue

            m = ROUND_START_RE.search(line)
            if m:
                r = int(m.group(1))
                if not multi_stream and last_round and r < last_round:
                    rep += 1            # round counter wrap -> next repeat
                    step = 0
                cur_round = r
                last_round = r
                continue

            m = PERCLIENT_RE.search(line)
            if m:
                cur_round = int(m.group(1))
                continue

            m = COST_RE.search(line)
            if m:
                cost_round = int(m.group(1))
                continue

            if 'per-group accuracy' in line:
                gm = GROUP_ACC_RE.search(line)
                if gm:
                    pairs = [(k.strip(), float(v))
                             for k, v in GPAIR_RE.findall(gm.group(1))]
                    try:
                        pairs.sort(key=lambda kv: float(kv[0]))
                    except ValueError:
                        pass
                    last_groups = pairs
                continue

            if 'ACC:' in line and 'disparity' not in line \
                    and 'per-group' not in line and '###' not in line:
                d = OrderedDict()
                for k, v in PAIR_RE.findall(line):
                    d[k] = float(v)
                if 'ACC' in d:
                    stats['metric_seen'] += 1
                    row = {'inv': inv, 'rep': rep, 'round': None,
                           'line': line_idx, 'groups': list(last_groups)}
                    row.update(d)
                    for k in d:
                        if k not in glob_cols:
                            glob_cols.append(k)
                    metric_cand.append(row)
                    last_groups = []
                continue

            if 'Checkpoint saved at round boundary' in line and not multi_stream:
                mk = REPEAT_ID_RE.search(line)
                if mk:
                    k = int(mk.group(1))
                    if k != rep:        # checkpoint-path validation
                        rep = k
                        step = 0
                continue

    # leftover unassigned metric candidates / unterminated block
    stats['metric_unpaired'] += len(metric_cand)
    if block_buf is not None:
        close_block(block_buf)
        block_buf = None

    return {'loss_rows': loss_rows, 'glob_rows': glob_rows,
            'loss_cols': loss_cols, 'glob_cols': glob_cols,
            'n_inv': dict(n_inv), 'stats': stats}


def choose_last_invocation(res):
    invs = set(res['n_inv'].keys())
    if not invs:
        return None
    with_g = [i for i in invs if res['n_inv'][i]['glob'] > 0]
    return max(with_g) if with_g else max(invs)


def write_loss_csv(rows, loss_cols, path):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['step', 'round', 'client', 'epoch', 'batch', 'repeat'] + loss_cols)
        for r in rows:
            w.writerow([r['step'], r['round'], r['client'], r['epoch'],
                        r['batch'], r['rep']] + [r.get(c, '') for c in loss_cols])


def write_global_csv(rows, glob_cols, path):
    max_g = max([len(r['groups']) for r in rows] + [0])
    gcols = ['acc_g%d' % i for i in range(max_g)]
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['communication_round', 'repeat'] + glob_cols + gcols)
        for r in rows:
            vals = [v for _, v in sorted(r['groups'], key=lambda kv: kv[0])]
            vals += [''] * (max_g - len(vals))
            w.writerow([r['round'], r['rep']] +
                       [r.get(c, '') for c in glob_cols] + vals)


def _agg_by_round(rows, col):
    """-> {round: [values...]} preserving multiplicity."""
    out = defaultdict(list)
    for r in rows:
        if col in r and r.get('round'):
            out[r['round']].append(r[col])
    return out


def _mean(vs):
    return sum(vs) / len(vs) if vs else float('nan')


def plot_losses(rows, loss_cols, tag, path, multi_stream):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    finite_pos = []

    if multi_stream:
        for ci, col in enumerate(loss_cols):
            c = colors[ci % len(colors)]
            per = _agg_by_round(rows, col)
            rounds = sorted(per)
            mean = [_mean(per[r]) for r in rounds]
            lo = [min(per[r]) for r in rounds]
            hi = [max(per[r]) for r in rounds]
            finite_pos += [v for vs in per.values() for v in vs if v > 0]
            ax.fill_between(rounds, lo, hi, color=c, alpha=0.15, linewidth=0)
            ax.plot(rounds, mean, color=c, lw=1.6, label=col)
        ax.set_xlabel('Communication round (all interleaved passes aggregated)')
    else:
        reps = sorted({r['rep'] for r in rows})
        per = {rp: [r for r in rows if r['rep'] == rp] for rp in reps}
        nmin = min(len(v) for v in per.values())
        for ci, col in enumerate(loss_cols):
            c = colors[ci % len(colors)]
            for rp in reps:
                ys = [r.get(col, float('nan')) for r in per[rp]]
                finite_pos += [v for v in ys if v > 0]
                ax.plot(range(len(ys)), ys, color=c, alpha=0.25, lw=0.7, zorder=1)
            mv = []
            for i in range(nmin):
                vals = [per[rp][i][col] for rp in reps if col in per[rp][i]]
                mv.append(_mean(vals))
            ax.plot(range(nmin), mv, color=c, lw=1.8, label=col, zorder=3)
        ax.set_xlabel('Training step (loss-block index within repeat)')

    ax.plot([], [], color='0.5', lw=0.7, alpha=0.6,
            label='single repeat (faint)' if not multi_stream
            else 'min-max band over passes')
    if finite_pos:
        try:
            ax.set_yscale('symlog', linthresh=max(min(finite_pos) * 0.5, 1e-6))
        except Exception:
            pass
    ax.set_ylabel('Loss value (symlog)')
    ax.set_title('%s: training loss terms%s' %
                 (tag, ' (multi-stream aggregate)' if multi_stream else
                  ' vs step (last invocation, %d repeat%s)' %
                  (len({r["rep"] for r in rows}),
                   's' if len({r["rep"] for r in rows}) > 1 else '')))
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2, loc='upper left', bbox_to_anchor=(1.01, 1.0))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_convergence(rows, glob_cols, tag, path, multi_stream):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    pairs = [('ACC', 'HM'), ('DEO', 'SPD')]
    pairs = [(a, b) for a, b in pairs if a in glob_cols or b in glob_cols]
    if not pairs:
        return False

    fig, axes = plt.subplots(len(pairs), 1, figsize=(10, 4.2 * len(pairs)),
                             sharex=True, squeeze=False)
    for axrow, (left, right) in zip(axes, pairs):
        ax = axrow[0]
        axr = ax.twinx()
        for axis, name, color in ((ax, left, 'tab:blue'),
                                  (axr, right, 'tab:orange')):
            if name not in glob_cols:
                continue
            per = _agg_by_round(rows, name)
            rounds = sorted(per)
            mean = [_mean(per[r]) for r in rounds]
            lo = [min(per[r]) for r in rounds]
            hi = [max(per[r]) for r in rounds]
            if multi_stream:
                axis.fill_between(rounds, lo, hi, color=color, alpha=0.18,
                                  linewidth=0)
                axis.plot(rounds, mean, color=color, lw=1.8, label=name)
                axis.set_title('%s: %s (left) / %s (right), '
                               'mean +- min/max over interleaved passes'
                               % (tag, left, right))
            else:
                reps = sorted({r['rep'] for r in rows})
                for rp in reps:
                    d = {}
                    for r in rows:
                        if r['rep'] == rp and name in r and r.get('round'):
                            d[r['round']] = r[name]
                    if d:
                        ks = sorted(d)
                        axis.plot(ks, [d[k] for k in ks], color=color,
                                  alpha=0.25, lw=0.8, zorder=1)
                axis.plot(rounds, mean, color=color, lw=1.9,
                          label='%s (mean over repeats)' % name)
                axis.set_title('%s: %s (left) / %s (right), '
                               'faint=single repeat' % (tag, left, right))
            axis.set_ylabel(name, color=color)
            axis.tick_params(axis='y', labelcolor=color)
        ax.grid(True, alpha=0.3)
    axes[-1][0].set_xlabel('Communication round')
    fig.suptitle('%s: global test metrics per communication round' % tag,
                 y=1.0, fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--log', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--outdir', default='/root/fl/analysis/curves')
    ap.add_argument('--multi-stream', action='store_true')
    ap.add_argument('--start-line', type=int, default=1)
    ap.add_argument('--no-plot', action='store_true')
    a = ap.parse_args()

    res = parse_log(a.log, multi_stream=a.multi_stream, start_line=a.start_line)
    os.makedirs(a.outdir, exist_ok=True)

    print('parse stats: %s' % res['stats'])
    print('invocations found: %s' %
          {k: v for k, v in sorted(res['n_inv'].items())})
    inv = choose_last_invocation(res)
    if inv is None:
        print('ERROR: no rows parsed from %s' % a.log)
        return 1
    loss_rows = [r for r in res['loss_rows'] if r['inv'] == inv]
    glob_rows = [r for r in res['glob_rows'] if r['inv'] == inv]
    reps_loss = sorted({r['rep'] for r in loss_rows})
    reps_glob = sorted({r['rep'] for r in glob_rows})
    rounds_glob = sorted({r['round'] for r in glob_rows})
    print('chosen invocation: %d | loss blocks: %d (repeats %s) | '
          'global rows: %d (repeats %s, rounds %s..%s)' %
          (inv, len(loss_rows), reps_loss[:8], len(glob_rows), reps_glob[:8],
           rounds_glob[0] if rounds_glob else -1,
           rounds_glob[-1] if rounds_glob else -1))
    print('loss columns: %s' % res['loss_cols'])
    print('global columns: %s' % res['glob_cols'])

    written = []
    lp = os.path.join(a.outdir, '%s_loss.csv' % a.tag)
    write_loss_csv(loss_rows, res['loss_cols'], lp)
    written.append(lp)
    if glob_rows:
        gp = os.path.join(a.outdir, '%s_global.csv' % a.tag)
        write_global_csv(glob_rows, res['glob_cols'], gp)
        written.append(gp)
    else:
        print('WARN: no global rows -> skip global csv')

    if not a.no_plot:
        if loss_rows and res['loss_cols']:
            pp = os.path.join(a.outdir, '%s_loss_curves.png' % a.tag)
            plot_losses(loss_rows, res['loss_cols'], a.tag, pp, a.multi_stream)
            written.append(pp)
        else:
            print('WARN: no loss rows -> skip loss plot')
        if glob_rows:
            cp = os.path.join(a.outdir, '%s_convergence.png' % a.tag)
            if plot_convergence(glob_rows, res['glob_cols'], a.tag, cp,
                                a.multi_stream):
                written.append(cp)
        else:
            print('WARN: no global rows -> skip convergence plot')

    for p in written:
        print('wrote %s (%d bytes)' % (p, os.path.getsize(p)))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
