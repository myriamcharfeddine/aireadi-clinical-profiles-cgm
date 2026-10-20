"""Redraw Figure 2 (CGM groups versus clinical profiles) into this folder from the saved outputs.

    python results/figures/figure_2_cgm_groups_versus_clinical_profiles/make_figure_2.py

Run the analyses first (bash run_all.sh). The drawing code is cgmprofiles.figures.figure_2, and the values
it plots are in source_data/.
"""
from cgmprofiles.figures import figure_2

if __name__ == "__main__":
    figure_2()
