# PrecisionRAG benchmark

Incomplete or one or more quality/latency targets not met.

Corpus: 100,000 passages. Dataset revision: `a47ee7aae8d7d466ba15f9f0bfac3b3681087b3a`.
Judge: `openai/gpt-oss-120b`. Run ID: `20261003T224627841693Z`.

| Mode | Context precision | Context recall | MRR@10 | Recall@5 | nDCG@10 | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| dense | 0.805 | 0.900 | 0.606 | 0.870 | 0.689 | 17.944 | 31.020 | 32.162 |
| hybrid_rerank_rrf | 0.900 | 0.900 | 0.715 | 0.910 | 0.775 | 458.379 | 563.505 | 621.546 |

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
| dense | 0.805 (0.698 to 0.897) | 0.900 (0.750 to 1.000) | 0.606 (0.534 to 0.678) | 0.870 (0.800 to 0.930) | 0.689 (0.630 to 0.747) |
| hybrid_rerank_rrf | 0.900 (0.813 to 0.966) | 0.900 (0.750 to 1.000) | 0.715 (0.650 to 0.782) | 0.910 (0.850 to 0.960) | 0.775 (0.722 to 0.829) |

## Improvement over dense, paired by query

Mean difference (mode minus dense) on the same queries, with its 95% interval and the share of bootstrap resamples in which the mode is ahead. Intervals that exclude zero are real at this sample size.

| Mode | Metric | Mean diff | 95% interval | P(better) | Queries better / worse |
|---|---|---:|---:|---:|---:|
| hybrid_rerank_rrf | context_precision | +0.094 | +0.019 to +0.196 | 1.00 | 10 / 2 (n=20) |
| hybrid_rerank_rrf | context_recall | +0.000 | +0.000 to +0.000 | 0.00 | 0 / 0 (n=20) |
| hybrid_rerank_rrf | mrr@10 | +0.109 | +0.035 to +0.180 | 1.00 | 35 / 17 (n=100) |
| hybrid_rerank_rrf | recall@5 | +0.040 | -0.020 to +0.100 | 0.87 | 7 / 3 (n=100) |
| hybrid_rerank_rrf | ndcg@10 | +0.086 | +0.029 to +0.141 | 1.00 | 35 / 18 (n=100) |

## By query category

MS MARCO tags each query with a type. Lexical anchors (numbers, names, codes) matter most for numeric, entity and person queries, so this is where hybrid retrieval should separate from dense.

| Category | Queries | dense MRR@10 | hybrid_rerank_rrf MRR@10 | dense Recall@5 | hybrid_rerank_rrf Recall@5 |
|---|---:|---:|---:|---:|---:|
| description | 64 | 0.622 | 0.709 | 0.875 | 0.906 |
| entity | 12 | 0.625 | 0.706 | 0.833 | 0.833 |
| location | 11 | 0.562 | 0.583 | 0.818 | 0.909 |
| numeric | 9 | 0.565 | 0.856 | 0.889 | 1.000 |
| person | 4 | 0.508 | 0.875 | 1.000 | 1.000 |

## LLM judge against human labels

RAGAS context precision is an LLM opinion. As a check, it is compared with the human relevance label for the same top-5: the judge should score higher when the labelled passage was retrieved.

| Mode | Judged | Spearman(precision, MRR@10) | Precision when labelled passage in top-5 | When missing |
|---|---:|---:|---:|---:|
| dense | 20 | 0.21 | 0.861 (18 q) | 0.308 (2 q) |
| hybrid_rerank_rrf | 20 | 0.33 | 0.924 (17 q) | 0.761 (3 q) |

Human labels are sparse (usually one passage per query), so the judge can legitimately score unlabelled passages as relevant; agreement in direction is what matters, not equality.

## Indexing

Measured pipeline time: 174.9 s; throughput: 571.7 passages/s. Under two hours: True. Index disk bytes observed: 1119137525 (0 if storage is outside this workspace).
