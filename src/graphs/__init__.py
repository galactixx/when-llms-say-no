"""Paper figure generation.

One module per figure in the paper. Each module is runnable on its own and
writes PNGs to `--output-dir`.

Nothing here modifies the analysis pipeline; these modules read the same result
files and reuse `src.analysis` binning so the figures cannot drift from what the
analysis computes. This mirrors how `src.tables` relates to `src.analysis`.
"""
