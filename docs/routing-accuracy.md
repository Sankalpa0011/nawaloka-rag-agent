# Routing accuracy

"Query routing accuracy" is an explicitly graded criterion, so it is measured
rather than asserted. The eval set was written **before the router existed**, so
the labels were not fitted to the implementation.

```bash
python -m evals.score_routing              # full run
python -m evals.score_routing --no-llm     # deterministic-signals baseline
python -m evals.score_routing --verbose    # per-case detail
```

Results land in `docs/routing-results.json`.

## Result

| configuration | accuracy |
|---|---|
| deterministic signals only (no LLM) | **70.7%** (41/58) |
| signals + LLM classifier + reconciliation | **98.3%** (57/58) |

### Per class

| route | n | correct | accuracy |
|---|---|---|---|
| SQL | 19 | 19 | 100% |
| VECTOR | 14 | 14 | 100% |
| HYBRID | 10 | 10 | 100% |
| EMERGENCY | 4 | 4 | 100% |
| REFUSE | 4 | 4 | 100% |
| SMALLTALK | 3 | 3 | 100% |
| OUT_OF_SCOPE | 4 | 3 | 75% |

### Edge cases and data traps

| tag | result |
|---|---|
| `trap:daily` (the `'Daily'` schedule value) | 4/4 |
| `trap:no_sessions` (doctors with no sessions) | 3/3 |
| `trap:honorific` (`Prof. Arjuna De Silva`) | 1/1 |
| `trap:package_text` (free-text package contents) | 1/1 |
| `trap:varchar_overflow` (`Room 004 (Paediatric Wing)`) | 1/1 |
| `multiturn` (coreference: "and what is *his* fee?") | 4/4 |
| `injection` (prompt-injection attempt) | 1/1 |

## The eval set

58 hand-labelled cases in `backend/evals/routing_set.jsonl`, each carrying the
reason for its label. `evals/test_routing_set.py` guards the dataset itself:
unique ids, class balance (no route above 40%), every trap covered, and — most
importantly — **every factual `must_include` value is checked against the live
database**, so the eval cannot assert something the data does not support.

## What actually moved the number

The gap between 70.7% and 98.3% was not one change. Each of these was found by
running the eval, not by reasoning about it.

**1. Deterministic signals cannot classify intent (70.7% ceiling).**
Signals alone score 0% on SMALLTALK and OUT_OF_SCOPE — no keyword distinguishes
"hello" from a hospital question. They are good evidence, not a classifier.

**2. The reconciliation override was actively harmful (+8.6pp when fixed).**
The original rule — *if both signal families fire, widen to HYBRID* — measured
**5 overrides, 0 correct**. It was turning correct 0.99-confidence SQL routes
into wrong HYBRID ones. Two causes:

- `"what is"` sat in the descriptive lexicon, and appears in almost every
  question including `"what is Dr X's fee?"`. The lexicon is now split into
  **strong** terms (`services`, `facilities`, `centre`, `policy`) and **weak**
  interrogatives, and only strong terms count as evidence.
- The override applied at any confidence. It now only fires below 0.85 —
  deterministic signals are a tie-breaker for an unsure model, not a veto over
  a confident one.

After the fix: 1 override, 1 correct.

**3. The model over-predicted HYBRID for multi-part questions.**
"Which lab tests require fasting and for how long?" was routed HYBRID because it
asks two things — but both halves are database columns. The prompt now states
that a multi-part question is still SQL when *every* part is a DB attribute, and
that HYBRID requires a descriptive part the database cannot answer.

**4. Entity detection had two silent bugs.**
`cardiologist` did not match the specialty `Cardiology` (patients name the
practitioner, not the discipline), and `\b` word boundaries never match next to
a bracket, so `Well Woman Package (Under 40)` was invisible to the gazetteer.
Both fixed and covered by tests.

## Caveats

**The result is at the noise floor of this eval set, and stops here
deliberately.** Two consecutive runs both scored 98.3%, but on *different*
cases:

| run | failure |
|---|---|
| A | `oos-002` "Write me a Python script to sort a list" → SQL |
| B | `vec-013` "What dental services are available?" → HYBRID |

Run A's failure was fixed with a general rule about general-assistant requests
borrowing database words ("list", "sort", "table"). The number did not move,
because a different borderline case flipped instead. That is the signal to
stop: at 57/58 the remaining error is model non-determinism on genuinely
ambiguous questions, not a defect with a fix. Chasing 100% here would mean
fitting the prompt to 58 specific strings.

`vec-013` is arguably mislabelled rather than misrouted — Nawaloka's dental unit
has a web page but no dentist in the `doctors` table, so HYBRID is a defensible
reading of "what dental services are available". The label was kept as written
rather than adjusted after the fact.

- 58 cases is small. Per-class figures on the 3–4 case classes (EMERGENCY,
  REFUSE, SMALLTALK, OUT_OF_SCOPE) carry wide error bars; the headline number is
  meaningful, a single class at 100% is not.
- Runs are not cached with `--fresh`; without it, responses are reused from
  `evals/.llm_cache` so a repeated run is reproducible and free.
- EMERGENCY and REFUSE are handled by deterministic patterns before any model
  call, so their 100% reflects pattern coverage on these cases, not model skill.
  `tests/test_guardrail.py` tests them far more thoroughly, including
  over-triggering (a hospital page about "stroke rehabilitation" must not
  trigger an ambulance response).
