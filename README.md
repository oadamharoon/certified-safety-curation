# Certified Safety Curation

Code for *Certified Safety Curation: Distribution-Free Guarantees for
Label-Efficient Safe Offline Reinforcement Learning*.

The method learns a state-only safety value from pairwise segment
preferences, scores whole trajectories with it, selects a subset with a
threshold calibrated by Learn-then-Test, and behavior-clones the
selection. The calibration yields a distribution-free (alpha, delta)
bound on the unsafe fraction of the selected training set, and refuses
to certify when the score cannot support one.

## Layout

    configs/     config.yaml (main, segment length 30) and the H = 10 / H = 50
                 variants used by the segment-length sweep. One config governs
                 every task; task blocks carry only dataset paths, the task
                 budget, and the sampling pool thresholds (see below).
    src/         model definitions (value ensemble, Gaussian policy, training
                 loops) and shared utilities (config loading, segmentation).
    pipeline/    the method, in order:
                   00b_dsrl_to_pickle      DSRL dataset -> trajectory pickle
                   01_segment_and_filter   trajectories -> active segments
                   03b_label_by_cost       segment pairs -> preference labels
                   04n_train_v_only        preferences -> value ensemble
                   04p_vfilter_bc          score, threshold at a fixed fraction, clone
                   04q_calibrated_vfilter  score, calibrate by LTT, clone or refuse
                   05_evaluate             100-episode evaluation
    analysis/    ablations, oracle controls, diagnostics, and the simulation
                 studies (guarantee resampling, e-process and pool-scaling
                 sims, margin probes, weighted-risk certificate).
    baselines/   CPL on the same preference data. Full-label baselines (CDT,
                 CPQ, COptiDICE) are trained with OSRL, external to this repo.
    experiments/ the queue scripts that orchestrated each campaign. They are
                 resume-safe: every job writes a done-marker, so a relaunch
                 skips completed work.
    paper/       result harvesting, table emission, and figure generation.
    legacy/      earlier scripts kept for provenance; not used by any reported
                 result.

## Experimental conventions

These hold uniformly across all twenty tasks; deviations were audited and
removed.

  preference pairs      1000 per task, segment length 30 (10 / 50 in the sweep)
  sampling pools        parent-trajectory cost quartiles: the safe pool is the
                        lowest quartile of episodic cost, the unsafe pool the
                        highest
  value ensemble        K = 3, two-layer MLP width 256, 300 epochs, batch 512
  behavior cloning      100 epochs, batch 512
  calibration           n = 200 labels, alpha = 0.25, delta = 0.1
  evaluation            100 episodes per checkpoint
  budgets               20 velocity, 25 navigation, 10 BulletSafetyGym
  seeds                 5 for headline configurations, 3 for analysis sweeps

## Reproducing a task end to end

    export SAFETY_VLM_TASK=pointgoal1_dsrl
    python pipeline/00b_dsrl_to_pickle.py
    python pipeline/01_segment_and_filter.py
    python pipeline/03b_label_by_cost.py
    SEED_OVERRIDE=0 python pipeline/04n_train_v_only.py
    SEED_OVERRIDE=0 MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 \
      python pipeline/04q_calibrated_vfilter.py
    python pipeline/05_evaluate.py --policy_file bc_calfilt_policy.pt

Data and trained artifacts are not tracked; every table and figure
regenerates from archived evaluation records via `paper/`.
