"""Redraw Figure 3 (Correspondence across glycemic strata) into this folder from the saved outputs.

    python results/figures/figure_3_correspondence_across_glycemic_strata/make_figure_3.py

Run the analyses first (bash run_all.sh). The drawing code is cgmprofiles.figures.figure_3, and the values
it plots are in source_data/.
"""
from cgmprofiles.figures import figure_3

if __name__ == "__main__":
    figure_3()
