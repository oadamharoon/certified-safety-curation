Tables that make_tables.py used to emit but the paper never inputs.

All five are superseded, and three of them (main, alpha, noise_policy) are built
on calfilt_ltt, a pre-standardization arm whose uncertified fallback rule the
code no longer has. On HalfCheetah that arm averages one certified seed with two
minimum-selection seeds and reports 1055 +/- 963, so inputting any of these would
put a known-bad number in the paper.

  main.tex            -> main_gated.tex (uses calfilt_csf, the deployed procedure)
  alpha.tex           -> Figure fig:alphacurve
  noise_policy.tex    -> prose plus Appendix app:extended
  oracle_main.tex     -> oracle_all.tex (all nine analysis tasks)
  vfilt_variants.tex  -> controls.tex

Kept for provenance. Their generators are commented out in make_tables.py.
