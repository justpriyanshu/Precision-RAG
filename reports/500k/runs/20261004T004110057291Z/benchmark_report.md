# PrecisionRAG benchmark

Incomplete or one or more quality/latency targets not met.

Corpus: 500,000 passages. Dataset revision: `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`.
Judge: `openai/gpt-oss-120b`. Run ID: `20261004T004110057291Z`.

| Mode | Context precision | Context recall | MRR@10 | Recall@5 | nDCG@10 | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.815 | 0.900 | 0.576 | 0.790 | 0.656 | 10.297 | 25.595 | 27.359 |
| hybrid_rerank_rrf | 0.834 | 0.850 | 0.671 | 0.850 | 0.733 | 135.744 | 168.294 | 194.593 |

## Measurement protocol

- Same frozen validation queries and qrels for every mode; 20 RAGAS queries per measured mode.
- 100 consecutive top-5 queries after warmup; result AND embedding caches disabled.
- Latency includes encoding, DB access, fusion and reranking; excludes HTTP transport and answer generation.
- IR is recalculated from saved top-10 rankings. RAGAS uses the first five of those same rankings.
- HNSW/index/vector parameters, hardware, package versions and corpus hashes are recorded in run_context.json.
- Both RAGAS means and latency percentiles are recalculated from saved per-query artifacts.

## Interpretation and limitations

This is a sampled MS MARCO QnA v2.1 corpus, not the official passage-ranking leaderboard. All selected validation evidence is reserved before train distractors are added. Labels are sparse; unjudged retrieved passages may be useful despite receiving zero binary relevance credit. Exact normalized duplicates are merged; semantic near-duplicates are not removed. Overlength selected evidence is excluded from evaluation, and training passages are clipped to 510 BGE tokens. The cross-encoder was trained on MS MARCO; this is an in-domain benchmark.

The filtered ablation uses the dataset query-type tag as an explicit scope constraint. It is a separate, favourable scoped workload, not evidence of general unfiltered improvement. Source/category arrays retain labels from observed duplicate occurrences. LLM judge variance and free-tier rate limits remain limitations. Missing scores are never replaced with estimates.

## Confidence intervals

95% percentile-bootstrap intervals of the mean over the frozen queries (2000 resamples). IR metrics use all latency/IR queries; RAGAS uses the judged subset.

| Mode | Context precision | Context recall | MRR@10 | Recall@5 | nDCG@10 |
|---|---:|---:|---:|---:|---:|
| dense | 0.815 (0.692 to 0.915) | 0.900 (0.750 to 1.000) | 0.576 (0.504 to 0.649) | 0.790 (0.710 to 0.870) | 0.656 (0.595 to 0.718) |
| hybrid_rerank_rrf | 0.834 (0.708 to 0.933) | 0.850 (0.700 to 1.000) | 0.671 (0.598 to 0.746) | 0.850 (0.780 to 0.910) | 0.733 (0.672 to 0.795) |

## Improvement over dense, paired by query

Mean difference (mode minus dense) on the same queries, with its 95% interval and the share of bootstrap resamples in which the mode is ahead. Intervals that exclude zero are real at this sample size.

| Mode | Metric | Mean diff | 95% interval | P(better) | Queries better / worse |
|---|---|---:|---:|---:|---:|
| hybrid_rerank_rrf | context_precision | +0.019 | -0.051 to +0.093 | 0.69 | 8 / 4 (n=20) |
| hybrid_rerank_rrf | context_recall | -0.050 | -0.150 to +0.000 | 0.00 | 0 / 1 (n=20) |
| hybrid_rerank_rrf | mrr@10 | +0.095 | +0.022 to +0.166 | 0.99 | 35 / 18 (n=100) |
| hybrid_rerank_rrf | recall@5 | +0.060 | +0.000 to +0.130 | 0.97 | 8 / 2 (n=100) |
| hybrid_rerank_rrf | ndcg@10 | +0.077 | +0.021 to +0.132 | 0.99 | 35 / 19 (n=100) |

## By query category

MS MARCO tags each query with a type. Lexical anchors (numbers, names, codes) matter most for numeric, entity and person queries, so this is where hybrid retrieval should separate from dense.

| Category | Queries | dense MRR@10 | hybrid_rerank_rrf MRR@10 | dense Recall@5 | hybrid_rerank_rrf Recall@5 |
|---|---:|---:|---:|---:|---:|
| description | 64 | 0.588 | 0.654 | 0.812 | 0.844 |
| entity | 12 | 0.608 | 0.663 | 0.750 | 0.750 |
| location | 11 | 0.543 | 0.553 | 0.727 | 0.818 |
| numeric | 9 | 0.500 | 0.856 | 0.778 | 1.000 |
| person | 4 | 0.542 | 0.875 | 0.750 | 1.000 |

## LLM judge against human labels

RAGAS context precision is an LLM opinion. As a check, it is compared with the human relevance label for the same top-5: the judge should score higher when the labelled passage was retrieved.

| Mode | Judged | Spearman(precision, MRR@10) | Precision when labelled passage in top-5 | When missing |
|---|---:|---:|---:|---:|
| dense | 20 | 0.06 | 0.873 (16 q) | 0.583 (4 q) |
| hybrid_rerank_rrf | 20 | 0.29 | 0.887 (15 q) | 0.673 (5 q) |

Human labels are sparse (usually one passage per query), so the judge can legitimately score unlabelled passages as relevant; agreement in direction is what matters, not equality.

## Indexing

Measured pipeline time: 867.6 s; throughput: 576.3 passages/s. Under two hours: True. Index disk bytes observed: 3218195691 (0 if storage is outside this workspace).
