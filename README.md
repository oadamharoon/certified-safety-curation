# Certified Safety Curation

Code and archived results for *Certified Safety Curation: Distribution-Free Guarantees for
Safe Offline Reinforcement Learning*.

The method learns a state-only safety value from pairwise segment preferences, scores whole
trajectories with it, selects a subset at a threshold calibrated by Learn-then-Test, and
behavior-clones the selection. Calibration yields a distribution-free (alpha, delta) bound on
the unsafe fraction of the selected training set, and refuses to certify when the score cannot
support one. What is certified is the training set, not the policy: the policy's cost is
measured and reported rather than bounded.

Every table the paper reports regenerates from the records here, and every number in the text
that is not a table cell is bound by an auditing script to the one expression that produces it.

<!-- AUTHORSHIP: this block is stripped from the anonymous archive built by
     tools/make_anonymous_zip.sh. Keep author, contact and arXiv details inside it. -->
**Paper.** arXiv:2609.12014.

**Authors.** Adam Haroon (aharoon@iastate.edu) and Cody Fleming, Iowa State University, Ames, IA, USA.

**Companion.** The same certificate applied to LLM fine-tuning data is *Clean Data, Unsafe Model:
Certified Safety Curation for LLM Fine-Tuning*, released separately.
<!-- END AUTHORSHIP -->

## Layout

    configs/     config.yaml (main, segment length 30) and the H = 10 / H = 50
                 variants used by the segment-length sweep. One config governs
                 every task; task blocks carry only dataset paths, the task
                 budget, and the sampling pool thresholds. Only the twenty tasks
                 the paper runs are listed.
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
    analysis/    ablations, oracle controls, diagnostics, the simulation studies
                 (guarantee resampling, e-process and pool-scaling sims, margin
                 probes, weighted-risk certificate), and the producers that write
                 the archived records under paper/data.
    baselines/   CPL on the same preference data, and the CPL pre-training stage
                 it calls. The full-label baselines (CDT, CPQ, COptiDICE) are
                 trained with OSRL; see third_party/.
    experiments/ the campaign drivers that orchestrated each run. They are
                 resume-safe: every job writes a done-marker, so a relaunch skips
                 completed work.
    tools/       the verification and lint scripts used to keep the results
                 honest, and the builder for the anonymous code archive.
    paper/       scripts/ regenerates every table and figure and runs the audit;
                 data/ holds the archived evaluation records they read;
                 figures/ the rendered figures. The paper source is not part of
                 this repository; a stale copy of it would be worse than none.
    runs/        the 2000-episode evaluation logs behind the deployment
                 certificate, and the selection summary the tables read. The rest
                 of the campaign output is raw training data and is not tracked.
    third_party/ the OSRL changes used to train the full-label baselines on our
                 certified selections: a subset_h5 config field and a gymnasium
                 import fix. OSRL and PREFINE themselves are upstream.

Every file here is reached by a result the paper reports.

## Regenerating the paper's tables and figures

From a clone, with no datasets and no trained artifacts:

    python paper/scripts/make_tables.py                        # all 27 tables
    python paper/scripts/make_figures.py                      # five figures
    FIG=alpha python paper/scripts/make_figures.py            # the alpha operating curve
    python paper/scripts/concept_figure.py                    # the schematic
    python paper/scripts/e_contamination_sweep.py             # controlled contamination
    PYTHONPATH=src python paper/scripts/landscape_layouts.py --replot   # safety landscape

All 27 tables come back byte-identical. Eight of the ten figures regenerate here and render
pixel-identically; their PDFs differ only in embedded timestamps. The other two, the Pareto
figure and the value-inspection figure, read trajectory pickles and checkpoints, so they need
the datasets and a trained ensemble. `--replot` on the landscape figure reuses the cached
panels rather than rolling out again.

## Checking the numbers

    python paper/scripts/verify_constants.py

This is the gate. It binds every number in the paper that is not a table cell to the one
expression that produces it, and refuses to let a check guard a sentence that is no longer
there. Two of its rules read the paper's LaTeX source, which is not part of this repository;
they announce a skip and the remaining 185 checks run. `verify_review_commitments.py` reads
that source in every rule, so it skips entirely unless the source is placed at
`paper/paper.tex`, which is also what the orphan guard needs.

`paper/scripts/completeness_check.py` goes further: it re-derives the archived records by
re-running their producers against the raw run outputs and checks provenance by file mtime, so
it needs the datasets and the campaign output and does not run from a clone alone.

## Rerunning the experiments

The scripts address four roots, resolved from the environment. The repository is located by
the `.csc-root` marker; the rest default beside it and are set when your layout differs:

    CSC_REPO       this repository, found by the .csc-root marker
    CSC_WORKSPACE  where the four roots below sit            (default: the repo's parent)
    CSC_WORK       datasets, checkpoints and per-task output (default: $CSC_WORKSPACE/datasets)
    CSC_RUNS       campaign output: selections, logs         (default: $CSC_WORKSPACE/runs, else $CSC_REPO/runs)
    CSC_OSRL       an OSRL checkout, for the full-label baselines
    PYTHON         the interpreter the shell drivers call    (default: python)

Most pipeline and analysis scripts read `config.yaml` from the working directory, so run them
from the directory that holds your datasets. One task, end to end:

    export SAFETY_VLM_TASK=pointgoal1_dsrl
    python pipeline/00b_dsrl_to_pickle.py
    python pipeline/01_segment_and_filter.py
    python pipeline/03b_label_by_cost.py
    SEED_OVERRIDE=0 python pipeline/04n_train_v_only.py
    SEED_OVERRIDE=0 MODE=ltt CAL_N=200 ALPHA=0.25 DELTA=0.1 \
      python pipeline/04q_calibrated_vfilter.py
    python pipeline/05_evaluate.py --policy_file bc_calfilt_policy.pt

## Data and models

Datasets, value-ensemble checkpoints, cloned policies and raw campaign output are not tracked:
they are large and regenerable, and the DSRL datasets are public. What is tracked is every
evaluation record the paper's numbers are computed from, the 2000-episode evaluation logs
behind the deployment certificate, and the selection summary the tables read. Everything the
paper reports regenerates from those.
