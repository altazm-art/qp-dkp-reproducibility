# Calculation and plotting provenance

The author retained the calculation archive and historical Python scripts
in the qp-DKP project folder. The consolidated numerical kernel and benchmark
checks are the current implementations of those calculations. Historical
sources recovered for this release include:

- qp_dkp_reproduce.py: original spectrum/contour/baseline data generator;
- qp_dkp_c4_slice.py: fixed-Sigma contour and slice;
- qp_dkp_c1_peaks.py and qp_dkp_c2_convergence.py: peak and cutoff checks;
- qp_dkp_b1_small_r.py: shifted small-r observables;
- qp_dkp_c3_high_tau.py: adaptive sums and logarithmic guides;
- qp_dkp_b3_ref24_compare.py: the printed single-parameter approximation;
- qp_dkp_relativistic_comparison.py and qp_dkp_first_order_relativistic.py:
  matched NR/DKP comparison, fixed-Sigma scans and first-order correction;
- prepare_revision.py: finite-temperature entropy/heat-capacity and extended
  plotting conventions.

Some early artwork was produced in Mathematica and the original notebooks
were not retained. This release supplies Python replacements for all 13
figures. Their data are recomputed from the stated model and checked against
the tabulated benchmarks. The plotting ranges and state-count/cutoff
conventions match the revised manuscript. The code labels the heat-capacity
normalization at unity as a continuum guide, preserving the discrete-periodic
qualification in the manuscript.

The release contains no reviewer reports, response letters, private Drive
links, original reference-paper PDFs, or manuscript source files. Computational
methods and validation records needed to reproduce the numerical results are
included directly in the documentation and code.
