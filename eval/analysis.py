"""Statistics on top of the saved per-query logs: no new model or judge calls.

Everything here is recomputed from ir_<tag>.jsonl and ragas_<tag>.json inside a run directory,
so `scripts.recalculate` can regenerate it at any time.
"""
from __future__ import annotations

import math

import numpy as np

from app.artifacts import read_json, read_jsonl
from eval.metrics import ir_metrics

IR_KEYS = ['mrr@10', 'recall@5', 'ndcg@10']
RAGAS_KEYS = ['context_precision', 'context_recall']
N_BOOT = 2000


def bootstrap_ci(values, n_boot=N_BOOT, seed=0):
    """Percentile bootstrap of the mean. Deterministic for a given seed."""
    a = np.asarray(list(values), dtype=float)
    if a.size == 0:
        return None
    rng = np.random.default_rng(seed)
    means = rng.choice(a, size=(n_boot, a.size), replace=True).mean(axis=1)
    return {'mean': float(a.mean()), 'low': float(np.percentile(means, 2.5)),
            'high': float(np.percentile(means, 97.5)), 'n': int(a.size)}


def paired_bootstrap(base, other, n_boot=N_BOOT, seed=0):
    """CI of mean(other - base) over the same queries, plus the share of resamples where it is > 0."""
    b, o = np.asarray(base, dtype=float), np.asarray(other, dtype=float)
    if b.size == 0 or b.size != o.size:
        return None
    diff = o - b
    rng = np.random.default_rng(seed)
    means = rng.choice(diff, size=(n_boot, diff.size), replace=True).mean(axis=1)
    return {'mean_diff': float(diff.mean()), 'low': float(np.percentile(means, 2.5)),
            'high': float(np.percentile(means, 97.5)), 'p_improved': float((means > 0).mean()),
            'queries_better': int((diff > 0).sum()), 'queries_worse': int((diff < 0).sum()), 'n': int(diff.size)}


def spearman(x, y):
    """Spearman rank correlation without scipy; average ranks for ties."""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty(len(v))
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        return r
    if len(x) < 3:
        return None
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def load_mode(directory, tag, categories=None):
    """Per-query rows for one mode: IR metrics from saved rankings, RAGAS scores if judged."""
    rankings = read_jsonl(directory / f'ir_{tag}.jsonl')
    ragas_path = directory / f'ragas_{tag}.json'
    ragas = {r['qid']: r for r in read_json(ragas_path)['rows']} if ragas_path.exists() else {}
    rows = []
    for r in rankings:
        ranked = [h['pid'] for h in r['hits']]
        relevant = set(map(str, r['relevant_pids']))
        rank = next((i + 1 for i, pid in enumerate(ranked[:5]) if str(pid) in relevant), None)
        row = {'qid': r['qid'], 'category': (categories or {}).get(r['qid'], 'unknown'),
               'human_rank_top5': rank, **ir_metrics(ranked, r['relevant_pids'])}
        if r['qid'] in ragas:
            row.update({k: ragas[r['qid']][k] for k in RAGAS_KEYS})
        rows.append(row)
    return rows


def analyze(directory, tags, categories=None):
    modes = {tag: load_mode(directory, tag, categories) for tag in tags}
    out = {'n_boot': N_BOOT, 'modes': {}, 'paired_vs_dense': {}, 'by_category': {}, 'judge_vs_human': {}}
    for tag, rows in modes.items():
        ci = {k: bootstrap_ci([r[k] for r in rows]) for k in IR_KEYS}
        judged = [r for r in rows if 'context_precision' in r]
        for k in RAGAS_KEYS:
            ci[k] = bootstrap_ci([r[k] for r in judged]) if judged else None
        out['modes'][tag] = ci
    base = modes.get('dense')
    if base:
        base_by_qid = {r['qid']: r for r in base}
        for tag, rows in modes.items():
            if tag == 'dense':
                continue
            paired = {}
            for k in IR_KEYS + RAGAS_KEYS:
                pairs = [(base_by_qid[r['qid']][k], r[k]) for r in rows
                         if r['qid'] in base_by_qid and k in r and k in base_by_qid[r['qid']]]
                paired[k] = paired_bootstrap([p[0] for p in pairs], [p[1] for p in pairs]) if pairs else None
            out['paired_vs_dense'][tag] = paired
    if categories:
        for tag, rows in modes.items():
            table = {}
            for cat in sorted({r['category'] for r in rows}):
                sub = [r for r in rows if r['category'] == cat]
                table[cat] = {'queries': len(sub), **{k: float(np.mean([r[k] for r in sub])) for k in IR_KEYS}}
            out['by_category'][tag] = table
    for tag, rows in modes.items():
        judged = [r for r in rows if 'context_precision' in r]
        if len(judged) < 3:
            continue
        found = [r['context_precision'] for r in judged if r['human_rank_top5'] is not None]
        missed = [r['context_precision'] for r in judged if r['human_rank_top5'] is None]
        out['judge_vs_human'][tag] = {
            'judged_queries': len(judged),
            'spearman_precision_vs_mrr': spearman([r['context_precision'] for r in judged],
                                                  [r['mrr@10'] for r in judged]),
            'mean_precision_when_human_passage_in_top5': float(np.mean(found)) if found else None,
            'mean_precision_when_human_passage_missing': float(np.mean(missed)) if missed else None,
            'queries_with_human_passage_in_top5': len(found), 'queries_without': len(missed)}
    return out


def fmt_ci(ci, digits=3):
    if not ci:
        return 'NOT MEASURED'
    return f"{ci['mean']:.{digits}f} ({ci['low']:.{digits}f} to {ci['high']:.{digits}f})"


def markdown(analysis, tags):
    lines = ['## Confidence intervals', '',
             f"95% percentile-bootstrap intervals of the mean over the frozen queries ({analysis['n_boot']} resamples). "
             'IR metrics use all latency/IR queries; RAGAS uses the judged subset.', '',
             '| Mode | Context precision | Context recall | MRR@10 | Recall@5 | nDCG@10 |', '|---|---:|---:|---:|---:|---:|']
    for tag in tags:
        ci = analysis['modes'][tag]
        lines.append('| ' + tag + ' | ' + ' | '.join(fmt_ci(ci.get(k)) for k in RAGAS_KEYS + IR_KEYS) + ' |')
    if analysis['paired_vs_dense']:
        lines += ['', '## Improvement over dense, paired by query', '',
                  'Mean difference (mode minus dense) on the same queries, with its 95% interval and the share of '
                  'bootstrap resamples in which the mode is ahead. Intervals that exclude zero are real at this sample size.', '',
                  '| Mode | Metric | Mean diff | 95% interval | P(better) | Queries better / worse |', '|---|---|---:|---:|---:|---:|']
        for tag, paired in analysis['paired_vs_dense'].items():
            for k in RAGAS_KEYS + IR_KEYS:
                p = paired.get(k)
                if p:
                    lines.append(f"| {tag} | {k} | {p['mean_diff']:+.3f} | {p['low']:+.3f} to {p['high']:+.3f} | "
                                 f"{p['p_improved']:.2f} | {p['queries_better']} / {p['queries_worse']} (n={p['n']}) |")
    if analysis['by_category']:
        lines += ['', '## By query category', '',
                  'MS MARCO tags each query with a type. Lexical anchors (numbers, names, codes) matter most for '
                  'numeric, entity and person queries, so this is where hybrid retrieval should separate from dense.', '']
        cats = sorted({c for t in analysis['by_category'].values() for c in t})
        lines += ['| Category | Queries | ' + ' | '.join(f'{t} MRR@10' for t in tags) + ' | '
                  + ' | '.join(f'{t} Recall@5' for t in tags) + ' |',
                  '|---|---:|' + '---:|' * (2 * len(tags))]
        for cat in cats:
            n = next((analysis['by_category'][t][cat]['queries'] for t in tags if cat in analysis['by_category'][t]), 0)
            mrr = [f"{analysis['by_category'][t].get(cat, {}).get('mrr@10', float('nan')):.3f}" for t in tags]
            rec = [f"{analysis['by_category'][t].get(cat, {}).get('recall@5', float('nan')):.3f}" for t in tags]
            lines.append(f'| {cat} | {n} | ' + ' | '.join(mrr) + ' | ' + ' | '.join(rec) + ' |')
    if analysis['judge_vs_human']:
        lines += ['', '## LLM judge against human labels', '',
                  'RAGAS context precision is an LLM opinion. As a check, it is compared with the human relevance label '
                  'for the same top-5: the judge should score higher when the labelled passage was retrieved.', '',
                  '| Mode | Judged | Spearman(precision, MRR@10) | Precision when labelled passage in top-5 | When missing |',
                  '|---|---:|---:|---:|---:|']
        for tag, j in analysis['judge_vs_human'].items():
            rho = 'n/a' if j['spearman_precision_vs_mrr'] is None else f"{j['spearman_precision_vs_mrr']:.2f}"
            a = 'n/a' if j['mean_precision_when_human_passage_in_top5'] is None else \
                f"{j['mean_precision_when_human_passage_in_top5']:.3f} ({j['queries_with_human_passage_in_top5']} q)"
            b = 'n/a' if j['mean_precision_when_human_passage_missing'] is None else \
                f"{j['mean_precision_when_human_passage_missing']:.3f} ({j['queries_without']} q)"
            lines.append(f"| {tag} | {j['judged_queries']} | {rho} | {a} | {b} |")
        lines += ['', 'Human labels are sparse (usually one passage per query), so the judge can legitimately score '
                  'unlabelled passages as relevant; agreement in direction is what matters, not equality.']
    return lines
