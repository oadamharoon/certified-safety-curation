# ICLR 2027 — run board

Updated 2026-08-18. Legend: `[x]` done · `[~]` in flight · `[ ]` not started.
Strikethrough = superseded or cancelled.

## A. Closing the standardization gap

- [x] **A1 — Stage 3b** · 466 arms · *complete 08/18 01:14*
  BC-Safe/BC-All, random/return/bottom-return controls, gated R50, α∈{0.10,0.40},
  preference-calibrated, tier-2, labels-only family (10 configs), CPL.
  117 job failures occurred and were all recovered by retry passes; 466/466 done markers.
- [x] **A2 — Rebuild certified selections** · 14 selections · *complete*
  Covers the 5 (task, α) pairs whose V-scores changed: PointGoal1 α0.25, CarRun α0.25,
  CarGoal2 α0.40, PointGoal2 α0.40, PointGoal1 α0.40.
  The other 5 tasks in the α=0.40 tables (HalfCheetah, Walker2d, Ant, Swimmer, CarGoal1)
  were proven unchanged by the determinism check, so their existing results stand.
- [x] **A3 — CDT reruns** · *complete 08/21 14:09* · 42/42, 0 failures
  5 truncated duplicates and 42 failed-run dirs moved to `runs/logs/a3cdt/_quarantine`.
  All 42 keepers verified at 40 progress rows. Harvested 08/21.
  The original estimate of 30 undercounted twice: it omitted PointGoal1 α=0.40
  (3 selections × 3 seeds = 9, required by `a40_draws.tex` and `compose_a40.tex`) and
  counted the CarRun echo as 3 when it has 2 certified selections × 3 seeds = 6.
  30 + 9 + 3 = 42.
  21 jobs died 08/18 on a launch-sequencing failure (queue started while the h5
  rebuild was still writing `next_observations`); all were redone after the h5
  files were validated, and there were no failures after the 08/19 relaunch.
- [x] **A4 — Operator arms on regenerated selections** · *complete 08/18 04:35*
  PointGoal1 3 sel × 4 intensities × 3 seeds, CarRun echo × 4.
  Logged jointly with A7 spillover as 114 arms.
- [x] **A5 — T2.6 aggregators** · 18 runs · *complete 08/23* · landed in the snapshot
- [~] **A5v2 — T3.2 AWR sweep rerun** · **90 runs** · relaunched 08/23 13:13 ·
  **11/90 at 08/23 19:00, ETA ~41 h (finishing 08/25 midday)**
  Runner `runs/scripts/stage_a5v2_t32.sh`, -P 7, LOGDIR `runs/logs/a5v2`.
  9 tasks x 5 configs x seeds {1,2}. Seed 0 is REUSED, not rerun: its 45 runs
  postdate every task's active_segments.pkl, utils/common.py is unchanged since
  07-29, no commit touched 04f in the window, and the SEED_OVERRIDE patch cannot
  alter seed 0 since the config default is already 0. Scoping this properly
  saved ~30 h against the 135-job version.
  **Two earlier passes are void.** (a) 08/21-08/23, 63 runs: `04f` ignored
  SEED_OVERRIDE, so all seeds were bit-identical. (b) 08/23 07:44, 135 runs,
  killed after 40 min: the fix had been applied to `certified-safety-curation`
  (the release mirror) instead of `vlm-with-cpl/new_data/scripts` (the live
  tree), so it ran unpatched code. Seed 1/2 artifacts from both are quarantined
  in `runs/logs/_degenerate_awr`.
  Verified before relaunch: lint CLEAN (5 checks), smoke PASS (80 s), seed
  sensitivity PASS (max weight diff 9.58e-02 vs 0.0 before), live and mirror in
  sync.
  Prereg: preregistration_labels_split.md, T3.2+T2.6 (2026-08-10).
  **Prediction 1 provisionally falsified** on PointGoal1 (beta=0.1, cost 15.7 vs
  budget 25). CarGoal2 and PointGoal2 sit 10% over so they could flip too.
  Contingency 3 (rewrite Section 5 and the corridor discussion) triggers only if
  this survives the rerun. Do not act before A5v2 lands.

- [x] **A6 — CPU analyses** · *complete*
  Guarantee validation (08/17), label-complexity extension (08/17), α-operating curve (08/18),
  score–cost/return correlations, margin probe + expansion, contamination→cost, coverage.
- [x] **A7 — H-sweep calibrated arms** · 54 arms · *complete 08/18 15:27*
- [~] **A8 — Harvest, regenerate, re-derive every claim**
  Done so far: 23 tables and all 8 included figures regenerated; full numeric audit of the
  text against the snapshot (see below). Must repeat once A3 and A5 land, since both
  change the composability numbers.

**A-total** ≈ 2–2.5 days, dominated by A3. A6 is CPU-only and overlapped with GPU stages.

## Guards against silent corruption

Every bug here was silent: the harvest printed a healthy summary while results
were dropped or duplicated. Run the tools; do not rely on reading carefully.

- `runs/scripts/lint_pipeline.py` — before launching any stage. 5 static checks:
  double-escaped regexes, trainers launched with SEED_OVERRIDE that ignore it,
  bash functions used in xargs subshells without `export -f`, done-markers in
  ephemeral dirs, and **live tree vs release mirror out of sync**. Currently CLEAN.
- `runs/scripts/smoke_test.sh` — before every sweep. Does it complete? ~80 s.
  Shrinks via SAFETY_VLM_CONFIG; env knobs like V_EPOCHS are read from config
  only by several scripts and are silently ignored.
- `runs/scripts/check_seed_sensitivity.sh` — **the gate a sweep actually needs.**
  Does the seed change the weights? ~3 min. Completing and varying by seed are
  different properties; the T3.2 sweep did the first perfectly for 34 h.
- `iclr2027/scripts/validate_snapshot.py` — after every harvest, before any table
  is regenerated. Nonzero exit gates the harvest.

**Which tree is live.** `vlm-with-cpl/new_data/` is the original working code and
is what every runner executes. `certified-safety-curation/` is the cleaned public
mirror. Fix the live tree first, then mirror. Lint check 5 enforces this.

**Audit, 08/23.** Fingerprinted every multi-seed checkpoint group on disk, 1003
groups: all have genuinely differing seeds. Only T3.2 was ever affected. Its 3
apparent exceptions were seeds 0 and 1 from 08-11 plus a seed 2 from 08-14 run
against a different code state, not seed sensitivity.

**Latent, not urgent.** Five live trainers still ignore SEED_OVERRIDE (04d, 04e,
04g, 04i, 04j, 04_train_cpl). No runner calls them and no degenerate arm came
from any of them. Fix before they are ever used in a sweep.

**Deferred.** `04f` writes `v_network.pt` without OUT_TAG, so concurrent jobs on
one task race on it. Nothing in the tree reads that file, and editing the live
script mid-sweep would split the run across two code states. Fix after A5v2.

**Fixed 08/23.** 17 shell scripts called `log_run` inside xargs subshells without
exporting it, so FAIL lines were never written anywhere. Past runs through those
scripts may hold unlogged failures; `validate_snapshot.py` check D is how to find
the resulting gaps.

## B. Never-started experiments

- [ ] **B1 — External comparator (T1.2)** · 0.5–2 days
  No OSIL or FISOR code present in the tree yet; starts with the availability check.
  Fallback is published numbers with a protocol note.
- [ ] **B2 — VLM labeler arm (T3.5)** · 1–2 days
  Infrastructure confirmed present: `analysis/02_render_frames.py`, `analysis/03_vlm_query.py`.
  Highest-value remaining item, the only experiment that *enters* the weak-oracle regime
  rather than modelling it.

## C. Optional, cheap, recommended

- [ ] **C1 — Pool-percentile sensitivity** · {20/80, 25/75, 33/67} on 3 tasks · ~~30 min~~ **needs a code change plus ~27 pipeline runs**
  The bottom/top quartile split is not exposed as a parameter anywhere, so this is
  not a 30-minute job. Deferred until A5v2 frees the machine.
  Defends the uniform quartile rule with a measured curve after the PointGoal1 drop.
- [ ] **C2 — Repo path portability** · ~~22 files~~ ~~64 files~~ → **105 files** · ~1–2 h
  *(re-counted 08-19: runs/scripts 24, analysis 24, experiments 23+7, iclr2027/scripts 13,
  paper 13. The 64 figure counted only part of the tree.)*
  Grew because the standardization runners added in the sync also hardcode the tree
  (33 `.py`, 31 `.sh`). Do after runs settle.

## Paper-side work completed 2026-08-18

- [x] Numeric audit of every "N of fifteen" claim against the regenerated snapshot.
      Five corrections applied: BC-Safe 9→12 (now a match, not a win), the false
      "strict subset of BC-Safe's failures" claim removed, fixed-top-25% 11→10,
      preference-calibration 5→4.
- [x] `interpretability.pdf` regenerated (was Jul 24); saliency claim re-verified exactly.
- [x] `landscape_pointgoal1_layouts.pdf` regenerated (was Jul 17) and given a producer
      script for the first time (`compose_layouts.py`, reset seeds 7/11/23).
- [x] Orphaned "binned Spearman ρ up to 0.68" replaced with a reproducible recomputation
      (`landscape_visited.py`): ρ = 0.88 binned, 0.61 raw, vs 0.09–0.27 frozen.
- [x] Rewrote the BC-Safe comparison around the 12-vs-12 tie, with the true failure sets.
- [x] Rewrote the H-sweep paragraph. Safety is 7/9 at every H; certification is 6/33
      at H=30, 3/27 at H=50, and 0/27 at H=10.
- [x] Applied the title change (dropped "Label-Efficient").
- [x] Response letter filled for all 38 comments (`paper/comments/response-letter.md`).
- [x] Full claim audit. Seven numeric defects found and corrected, listed below.
- [x] Notation disambiguated. The clip is now $\kappa$, contamination is $u_j$, and
      hypergeometric population counts are $U_j$, so $c$ means only the cost function
      and $K$ only the ensemble size.
- [x] Normalized-cost table moved to the appendix, raw kept primary.
- [x] 37 colon splices removed (Fleming C004).
- [x] H-sweep mechanism resolved. Resampled 2000 calibration draws per task and seed
      at each H. Certification rate declines 0.204 / 0.157 / 0.102 at H = 50 / 30 / 10,
      tracking mean purity margin (-0.002 / -0.015 / -0.072). Margin predicts rate with
      Spearman 0.988 across 27 task-length cells, replicating the law under a design
      change. The earlier "collapse to 0 of 27" was a three-seed artifact; PointGoal1's
      true H = 10 rate is 0.69.
- [x] Bootstrap CI on the label-complexity exponent: 1.57, 95% CI [1.32, 1.65].
- [x] BC-Safe margin surfaced: lower cost on 8 of the 11 jointly-safe tasks.
- [ ] **Main text is 13 pages against a 9-page limit.** Needs a trim pass.

## D. Data-integrity fixes (2026-08-19)

Found while auditing figures and tables. All are code or data defects, not
missing science, and none moved the headline count of 11 of 15.

- [x] **Harvester merged two configs.** Metadata for the pre-standardization
      `calfilt_lttv2` tag was routed to the same key as `calfilt_ltt`, so glob
      order decided which won. Four seed-cells disagreed on `certified` itself.
      Values were already routed correctly; only metadata was wrong.
      **Consequence:** the certifying set is four tasks (HalfCheetah, Ant,
      CarGoal1, PointGoal1), not three, and PointGoal1 certifies on 2 of 5
      seeds, not 4. The paper says "three tasks" in five places and still
      needs updating.
- [x] **Table generator hardcoded the certifying tasks**, so Table 3 marked
      CarGoal1 with a certification dagger it had lost. Now derived from data.
- [x] **`cond_viol` table printed five duplicated Bullet rows** with conflicting
      values, from a stale August 4 file re-emitted beside the current one.
- [x] **Pareto figure plotted the wrong values** (Ours 10 instead of 11, CDT 5
      instead of 7) because it used raw `calfilt_ltt` rather than the deployed
      rule, and CDT's default target rather than its safest sweep target.
- [x] **Figure 5 caption described an axis that no longer exists** (a continuous
      label-fraction axis, from before the supervision reframe).
- [x] **Figure 4 lost its radial panel** when I rebuilt it after the scratchpad
      wipe; rebuilt with all four panels and a producer script that did not
      previously exist.
- [x] **`alpha_curve.pdf` was stale** (August 4) and its Bullet half read a
      pre-standardization file; both regenerated.
- [x] **`num_pairs` in the configs read 3000** while every reported task used
      1000, so the repo would not have reproduced the runs.
- [x] **`calfilt_csf` completed on the three missing tasks** (15 runs, 0 failures).
      It is now uniform at five seeds across all fifteen tasks and is the single
      deployed config. `calfilt_ltt` (July, obsolete fallback rule) is retired
      from the tables.
- [x] **Gated variant re-run** (15 runs, 0 failures) so it pairs with the
      deployed config. All three certifying tasks now agree on certified seeds
      and thresholds to the digit; PointGoal1 previously disagreed.
- [x] **Metadata harvest made structural.** Tags are parsed generically instead
      of matched against a fixed list, and unmatched tags are printed rather
      than dropped. The old list silently discarded metadata for six tags,
      including the H-sweep arms.
- [x] **Operator-grid contamination derived from data.** The hardcoded literals
      had drifted from the regenerated ensembles by up to 0.037.
- [x] **Five orphan table files removed**; every table the paper inputs is
      present and current.
- [x] **Pareto and alpha-curve figures repointed** at the deployed config and
      the regenerated Bullet data.

### Verified after the completion runs
Deployed `calfilt_csf`, five seeds, fifteen tasks: **11 of 15 safe**, failures
Hopper, Swimmer, PointButton2, PointCircle2. **Three certifying tasks**
(HalfCheetah 3/5, CarGoal1 1/5, PointGoal1 2/5), which is what the paper claims.
All ten count assertions in the text match the data. Specific values re-checked
(Swimmer 19.5 / 72.2 / 44.7, Hopper 22.9 and ungated 148.3, Walker2d 5.4).
Paper compiles with 0 errors and 0 undefined references.

### Known remaining, all A3-dependent
- `compose.tex`, `compose_a40.tex`, `a40_draws.tex` still hold pre-standardization
  CDT numbers until A3 lands.
- Re-run A8 (regenerate every table and figure, re-audit every claim) afterwards.
- Main text is 13 pages against a 9-page limit; trim after results are final.
