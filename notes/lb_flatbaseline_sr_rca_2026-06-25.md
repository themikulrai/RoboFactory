# LiftBarrier flat-baseline SR collapse — root-cause analysis (2026-06-25)

**Question:** why did the flat-baseline LB pi0.5 (decentralized) SR drop "suddenly" to ~2/60, and how to prevent it.
**Method:** 2 dynamic workflows (29 agents): criterion decomposition on the SAME 120 contact traces, contact-sensor
reliability, historical anchor, git criterion-archaeology, 22 video ground-truths, devil's-advocate refutation, synthesis.
**Data:** full-horizon (LB_EVAL_NO_TERMINATE) contact traces, ckpt 14999, seeds 20000-20059, prompt "lift the barrier using
two robot arms". WS=lb_ws_flatbaseline_15990490, WC=lb_wc_flatbaseline_15990491.

## Verdict: MIXED — real coordination failure + a median-aggregation undercount artifact

### The drop happened in TWO stages (identical ckpt/seeds/prompt/sha; ONLY criterion code changed)
| Criterion (same rollouts) | WS | WC | counts |
|---|---|---|---|
| #1 Official: center-z>0.15, instantaneous, early-terminate, grasp-blind | 56/60 | 49/60 | flings/tips/one-arm all pass (INFLATED) |
| #3 Geometric: both-ends-held (TCP+gripper-closed), sustained, full-horizon | 18/60 | 27/60 | removes flings/non-held |
| #4 Shipping: median of min(f0,f1) over lifted frames > 1N | 2/60 | 2/60 | requires sustained two-arm LOAD |
| consec-window (>=8 consec both-arms>1N + level): the *fix* | 16/60 | 25/60 | isolates the brief co-lift event |
| **video-verified TRUE cooperative SR** | **~13/60** | **~7/60** | deliberate co-lift genuinely occurred |

Decompose ladder proved height(0.15->0.25)/center-vs-ends/sustain knobs contribute ~0 (all leave geometric at 57/54 on these
traces). The collapse is entirely the two-arm requirement.

### Stage 1 (56->18 ws) is REAL and correct
Flat policy genuinely lifts ONE-ARMED in ~90% of seeds (cantilever WS 55/60, WC 49/60): hooks one end, cantilevers the bar to
0.6-0.84m on one arm, other arm never bears load. Triple-confirmed: (a) force bimodal — 71.7% of arm-frames read EXACTLY 0N
vs load mode ~7.6N median, clean valley at 1N; (b) ZERO lifted-with-both-arms-near-zero seeds (no dead-sensor false zeros),
load-bearing arm FLIPS per seed (sensor alive on both); (c) cantilever bars tilt 0.19-0.28m (one-end pivot) vs genuine holds
level at 0.02-0.04m. Old 56/60=93% was an inflated upper bound (instantaneous grasp-blind center-z counts one-arm lifts/flings).

### Stage 2 (18->2) is largely a MEDIAN-AGGREGATION ARTIFACT
Shipping criterion = median of min(f0,f1) over ALL lifted frames. The policy does a brief (~8-12 frame) genuine two-arm
co-lift, THEN transitions to a one-arm dangling carry for hundreds of frames (totLifted 180-430). The long one-arm tail drags
the median of min-force below 1N -> scores 0 even though a real co-lift happened. The 2 "successes"/cam are simply the
SHORTEST-lifted episodes ("lift then set down" beats "lift then hold aloft" — inverts quality order).

### Video verification of 10 near-miss (consec-pass, median-fail) seeds: 5/10 deliberate, CAMERA-ASYMMETRIC
- WS near-misses 4/5 deliberate two-arm holds (genuine co-lift, then decays to one-arm).
- WC near-misses 1/5 deliberate (4/5 = transient brushes that pop the bar then drop/dangle one-arm).
- NONE sustain: all deliberate holds are ~8-12 frames then -> one-arm carry (7/10) or drop (3/10).
- => consec 16/25 is the right DIRECTION but OVER-COUNTS WC (admits brushes). True cooperative SR: **WS ~13/60 (9-16), WC ~7/60 (4-10)**.

## Why it "suddenly" collapsed = measurement governance failure
- Criterion mutated 4x (2025-07-29 -> 2026-06-24) with NOTHING stamping which version produced a given SR.
- Shipped results JSON literally records success=0/60 (LB_EVAL_NO_TERMINATE env path) while the true 2/60 lives in an
  OFF-REPO offline scorer (score_contact.py). The two are never reconciled in any artifact.
- Anchors were non-comparable: "93%" = inflated instantaneous criterion; memory "32/60" = a Diffusion-Policy run (wrong
  policy family). Honest comparable before = geometric 18/27; faithful after = ~13/7 -> NOT a catastrophic regression.
- Run-to-run sim noise ~2-3/60 — same order as the shipping 2/60, so 2/60 is statistically a handful near floor.

## Prevention (ranked)
1. (low) Criterion version-stamp in every results JSON + npz trace: {crit_sha, mode, DZ, K_MIN, TAU, aggregation, window,
   no_terminate}. Refuse to render SR without it; reconcile env-path success with the offline scorer in ONE artifact.
2. (low-med) Standing spectrum on every eval: report official / geometric / consec-quality / median side-by-side so a drop
   is instantly attributable to the knob that moved.
3. (med) FIX the aggregation: replace median-over-all-lifted with a consecutive-window quality criterion (>=K consec frames
   both-arms>TAU + level + lifted) PLUS a grasp gate, AND require the hold not to immediately drop (kills WC brushes).
   Compute any median over the HOLD WINDOW, not all lifted frames.
4. (med-high) Single-source criterion registry: one module imported by env.evaluate(), the offline scorer, AND the datagen
   keep-gate (they currently drift; db84dae's 0.32->0.45 datagen lift was mistaken for a success-DZ change).
5. (low) Threshold provenance: document H/TAU/K + aggregation choice with this RCA's calibration (bimodal valley, tilt sep).

## Caveats / problems with this analysis
- True-SR point estimates rest on a 5-per-camera video sample of the near-miss pool (extrapolated by deliberate-fraction); ±3-4 seeds.
- Several video verdicts were medium-confidence; tiled-multiview crop quality limited.
- The full-horizon NO_TERMINATE mode itself biases policy comparison (penalizes "lift and keep aloft"); any criterion built on
  it inherits that bias unless success is detected at the FIRST qualifying hold window (offline early-terminate semantics).
- Old/geometric anchors are themselves upper bounds (LB success() instantaneous-center-z false-positive still applies to them).
