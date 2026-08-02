# Retrieval calibration

Measured on the 140-chunk corpus with `BAAI/bge-small-en-v1.5` (dense) +
`Qdrant/bm25` (sparse), RRF-fused, reranked by `ms-marco-MiniLM-L-6-v2`.
Scores are the sigmoid of the cross-encoder logit.

Reproduce with:

```bash
python -m app.retrieval.search "heart centre services" -k 20 --floor 0
```

## Why there is a score floor at all

If the top-k is returned unconditionally, an out-of-scope question ("what is the
capital of France") still yields five confidently-formatted hospital passages,
and the answer model will use them. Abstention has to be a retrieval decision,
not something the LLM is asked to notice.

## Score distributions

The reranker is sharply discriminative — relevant passages score near 1.0,
irrelevant ones collapse to ~0.000 with almost nothing in between.

| query | top | 2nd | 3rd | 5th | 10th |
|---|---|---|---|---|---|
| what facilities does the neuro centre have | 0.998 | 0.985 | 0.935 | 0.792 | 0.002 |
| physiotherapy rehabilitation | 0.997 | 0.997 | 0.989 | 0.003 | 0.000 |
| ambulance emergency contact number | 0.995 | 0.984 | 0.940 | 0.097 | 0.001 |
| international patient services | 0.981 | 0.887 | 0.816 | 0.380 | 0.030 |
| what awards has the hospital won | 0.981 | 0.958 | 0.856 | 0.000 | 0.000 |
| dental services | 0.972 | 0.561 | 0.464 | 0.414 | 0.001 |
| how do I book a channeling appointment | 0.962 | 0.878 | 0.005 | 0.004 | 0.000 |
| heart centre services | 0.912 | 0.898 | 0.894 | 0.205 | 0.001 |
| do you have a blood bank | 0.529 | 0.060 | 0.000 | 0.000 | 0.000 |
| do you test for vitamin D | 0.080 | 0.000 | 0.000 | 0.000 | 0.000 |
| **what is the capital of France** | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 |
| **how do I reset my wifi router** | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 |
| **recipe for chocolate cake** | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 |
| **stock price of Tesla** | **0.000** | 0.000 | 0.000 | 0.000 | 0.000 |

## Chunks surviving each absolute floor

| query | 0.001 | 0.005 | 0.01 | **0.02** | 0.05 | 0.10 | 0.15 |
|---|---|---|---|---|---|---|---|
| heart centre services | 8 | 6 | 6 | **5** | 5 | 5 | 5 |
| what facilities does the neuro centre have | 11 | 9 | 8 | **8** | 8 | 8 | 7 |
| do you have a blood bank | 2 | 2 | 2 | **2** | 2 | 1 | 1 |
| ambulance emergency contact number | 9 | 7 | 7 | **7** | 5 | 4 | 3 |
| how do I book a channeling appointment | 6 | 3 | 2 | **2** | 2 | 2 | 2 |
| do you test for vitamin D | 1 | 1 | 1 | **1** | 1 | **0** | **0** |
| international patient services | 18 | 15 | 14 | **10** | 7 | 7 | 7 |
| physiotherapy rehabilitation | 5 | 4 | 4 | **4** | 4 | 4 | 4 |
| dental services | 10 | 8 | 8 | **7** | 6 | 6 | 6 |
| what awards has the hospital won | 4 | 4 | 4 | **4** | 4 | 4 | 4 |
| *4 out-of-scope queries* | 0 | 0 | 0 | **0** | 0 | 0 | 0 |

## Decision: absolute floor = 0.02

- Every out-of-scope query yields **zero** chunks at any floor, so abstention is
  robust — the separation between "relevant" and "irrelevant" is three orders of
  magnitude, not a marginal call.
- 0.02 keeps 2–10 chunks on real queries. The initial guess of 0.15 was too
  strict: it silently reduced `do you test for vitamin D` to zero results, which
  would have surfaced as "we don't offer that test" for a test the hospital
  does offer.

## Rejected: relative floor (keep score ≥ ratio × top score)

A relative floor looks attractive given how steep the distributions are, but it
**inverts on exactly the queries that matter most**. When every candidate scores
0.000, `ratio × top` is also 0, so the whole candidate set passes:

| query | ratio 0.02 | 0.05 | 0.10 | 0.20 |
|---|---|---|---|---|
| heart centre services | 5 | 5 | 5 | 5 |
| **what is the capital of France** | **20** | **20** | **20** | **20** |
| **recipe for chocolate cake** | **20** | **20** | **20** | **20** |

Out-of-scope queries would admit all 20 candidates instead of none. Rejected.

## Latency: where the time actually goes

Per-stage, warm, on an 8-core CPU:

| stage | time |
|---|---|
| dense embed (bge-small) | 7–40 ms |
| sparse embed (BM25) | <1 ms |
| Qdrant hybrid query + RRF | 64–80 ms |
| **cross-encoder rerank (20 docs × 1200 chars)** | **~1,800 ms** |

Reranking is essentially the whole cost. Two bugs were fixed along the way:

- `get_client()` constructed a **new `QdrantClient` per call**, adding ~700 ms
  of connection setup to every search. Now cached process-wide — the query
  itself dropped to ~70 ms.
- ONNX thread count was left to the runtime default; it is now set explicitly
  to `cpu_count - 1`.

### Rerank cost scales linearly in both dimensions (ms)

| docs | 400 ch | 600 ch | 800 ch | 1200 ch | 2000 ch |
|---|---|---|---|---|---|
| 8 | 158 | 225 | 357 | 637 | 1104 |
| 12 | 356 | 420 | 568 | 1067 | 1526 |
| 16 | 416 | 734 | 765 | 1213 | 1915 |
| 20 | 518 | 763 | 1024 | 1810 | 2463 |

### Truncating documents was measured, and rejected

Clipping chunk bodies before reranking is the obvious speed lever, so it was
tested against full-length (4000-char) reranking on 12 queries:

| clip | mean top-5 agreement | rank-1 identical |
|---|---|---|
| 400 | 79% | 8/12 |
| 600 | 79% | 8/12 |
| 800 | 88% | 9/12 |
| **1200** | **100%** | **11/12** |
| 4000 | 100% | 11/12 |

1200 characters is the point at which truncation becomes free — identical
results to full-length reranking. Below it, quality degrades sharply and
non-uniformly: `do you have a blood bank` loses its **entire** top-5 (0%
overlap) at 800 chars and below, because the distinguishing text sits late in
the chunk. Speed was therefore taken from the candidate pool
(`RETRIEVAL_FUSION_LIMIT` 20 → 12, ~1.8 s → ~1.1 s) rather than from document
length, since candidates ranked 13–20 by RRF almost never survive the score
floor anyway.

## Note on the lab catalogue

`do you test for vitamin D` peaks at 0.080 — much lower than prose queries.
Catalogue chunks are lists of test names, which cross-encoders score weakly
against natural-language questions. It clears the 0.02 floor, and for the 17
tests present in the relational database the SQL path is authoritative anyway;
the catalogue only answers availability for the remaining ~1,240.
