"""Redraw Figure 1 (CGM outcomes across clinical profiles) into this folder from the saved outputs.

    python results/figures/figure_1_cgm_outcomes_across_clinical_profiles/make_figure_1.py

Run the analyses first (bash run_all.sh). The drawing code is cgmprofiles.figures.figure_1, and the values
it plots are in source_data/.
"""
from cgmprofiles.figures import figure_1

if __name__ == "__main__":
    figure_1()
