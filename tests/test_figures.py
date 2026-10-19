"""Figures: every plotted estimate equals its source table; labels and colours follow the conventions."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgmprofiles.config import get_config, results_dir

ROOT = Path(__file__).resolve().parents[1]
STEMS = ["fig1_forward", "fig2_reverse", "fig3_strata", "figS1_temporal"]
AGG = results_dir("aggregate")


def plotted(stem: str) -> dict:
    path = results_dir("figures") / f"{stem}_plotted.json"
    if not path.exists():
        pytest.skip("figures not generated yet")
    return json.loads(path.read_text())


@pytest.mark.parametrize("stem", STEMS)
def test_figure_files_exist(stem):
    plotted(stem)
    for suffix in (".pdf", ".png"):
        assert (results_dir("figures") / f"{stem}{suffix}").stat().st_size > 10_000


def test_fig1_eta_matches_kruskal_wallis():
    values = plotted("fig1_forward")
    kw = pd.read_csv(AGG / "forward_kruskal_wallis.csv").query("canonical_stratum == 't2d_oral_non_insulin'").set_index(["window", "metric"])
    for key, panel in values.items():
        window, metric = re.findall(r"'([^']+)'", key)
        assert panel["eta_squared"] == pytest.approx(kw.loc[(window, metric), "eta_squared"], abs=1e-12)
        assert [panel[c]["n"] for c in ("C1", "C2", "C3")] == [192, 111, 155]


def test_fig2_panels_match_sources():
    values = plotted("fig2_reverse")
    counts = pd.read_csv(AGG / "reverse_primary_contingency.csv")[["C1", "C2", "C3"]].to_numpy()
    assert np.array_equal(np.array(values["A"]["counts"]), counts)
    fr = pd.read_csv(AGG / "forward_vs_reverse_eta_squared.csv")
    hba1c = fr[fr.hba1c_flag].iloc[0]
    assert values["B"]["HbA1c"] == pytest.approx([hba1c.eta_squared, hba1c.ci_low, hba1c.ci_high], abs=1e-12)
    b6 = pd.read_csv(AGG / "pooled_agreement.csv").query("metric == 'ARI' and target == 'diagnostic stratum (4)'").sort_values("pooled_k")
    for got, want in zip(values["D"]["diagnostic stratum (4)"], b6[["value", "ci_low", "ci_high"]].to_numpy().tolist()):
        assert got == pytest.approx(want, abs=1e-12)


def test_fig3_specificity_matches_source():
    values = plotted("fig3_strata")["C"]
    spec = pd.read_csv(AGG / "strata_specificity.csv").query("variant == 'primary (T2D scaler, all)' and metric == 'coefficient_of_variation_pct'")
    for r in spec.itertuples():
        key = str((r.canonical_stratum, r.window))
        assert values[key] == pytest.approx([r.mean_diff_c2_minus_rest, r.diff_ci_low, r.diff_ci_high], abs=1e-12)


def test_figS1_uses_primary_design():
    values = plotted("figS1_temporal")
    c = pd.read_csv(AGG / "temporal_c2_contrast.csv").query("design == 'days3'").iloc[0]
    assert values["C"][0] == pytest.approx([c.A_difference, c.A_ci_low, c.A_ci_high], abs=1e-12)


def test_labels_and_colours_follow_conventions():
    source = "\n".join(p.read_text() for p in (ROOT / "src/cgmprofiles/plotting").glob("*.py"))
    assert "T2D oral" not in source and "#FF0000" not in source.upper()
    cfg = get_config().colors
    strata, profiles = set(cfg.strata.raw.values()), set(cfg.profiles.raw.values())
    assert len(strata) == 4 and not strata & profiles


PRINT_STEMS = ["fig1_forward_print", "fig2_reverse_print", "fig3_strata_print"]


def plotted_print(stem: str) -> dict:
    path = results_dir("figures", "print") / f"{stem}_plotted.json"
    if not path.exists():
        pytest.skip("print figures not generated yet")
    return json.loads(path.read_text())


@pytest.mark.parametrize("stem", PRINT_STEMS)
def test_print_figure_files_exist(stem):
    plotted_print(stem)
    for suffix in (".pdf", ".png"):
        assert (results_dir("figures", "print") / f"{stem}{suffix}").stat().st_size > 10_000


def test_print_figures_show_the_same_values_as_the_manuscript_figures():
    main1, print1 = plotted("fig1_forward"), plotted_print("fig1_forward_print")
    for key, panel in print1.items():
        for c in ("C1", "C2", "C3"):
            assert panel[c]["n"] == main1[key][c]["n"] and panel[c]["median"] == pytest.approx(main1[key][c]["median"], abs=1e-12)
    main2, print2 = plotted("fig2_reverse"), plotted_print("fig2_reverse_print")
    assert print2["A"]["counts"] == main2["A"]["counts"]
    for metric in ("balanced_accuracy", "macro_auroc"):
        assert print2["C"][metric]["cgm"] == pytest.approx(main2["C"][metric]["cgm"], abs=1e-12)
        assert print2["C"][metric]["both"] == pytest.approx(main2["C"][metric]["both"], abs=1e-12)
    assert print2["D"]["diagnostic stratum (4)"] == main2["D"]["diagnostic stratum (4)"]
    main3, print3 = plotted("fig3_strata"), plotted_print("fig3_strata_print")
    for key, value in print3["C"].items():
        assert value == pytest.approx(main3["C"][key], abs=1e-12)
    for key, value in print3["A"].items():
        assert value == pytest.approx(main3["A"][key], abs=1e-12)


ALTERNATE_STEMS = ["fig2_correspondence", "figS_reverse_supplement"]


def plotted_alternate(stem: str) -> dict:
    path = results_dir("figures", "alternates") / f"{stem}_plotted.json"
    if not path.exists():
        pytest.skip("alternate figures not generated yet")
    return json.loads(path.read_text())


@pytest.mark.parametrize("stem", ALTERNATE_STEMS)
def test_alternate_figure_files_exist(stem):
    plotted_alternate(stem)
    folder = results_dir("figures", "alternates")
    for suffix in (".pdf", ".png"):
        assert (folder / f"{stem}{suffix}").stat().st_size > 10_000
    assert (folder / f"{stem}_caption.tex").read_text().startswith("\\caption{")


def test_alternate_figures_show_the_same_values_as_fig2():
    main2, corr, supp = plotted("fig2_reverse"), plotted_alternate("fig2_correspondence"), plotted_alternate("figS_reverse_supplement")
    assert corr["A"]["counts"] == main2["A"]["counts"]
    assert corr["C"]["cgm"] == pytest.approx(main2["C"]["macro_auroc"]["cgm"], abs=1e-12)
    assert corr["C"]["both"] == pytest.approx(main2["C"]["macro_auroc"]["both"], abs=1e-12)
    hba1c = pd.read_csv(AGG / "forward_vs_reverse_eta_squared.csv").query("hba1c_flag").iloc[0]
    assert corr["B"]["hba1c_percent_baseline"] == pytest.approx([hba1c.eta_squared, hba1c.ci_low, hba1c.ci_high], abs=1e-12)
    r2 = pd.read_csv(AGG / "reverse_prediction.csv").query("direction == 'CGM to clinical variable' and feature_set == 'extended' and window == 'window_full'")
    hb = r2.set_index("target").loc["hba1c_percent_baseline"]
    assert corr["D"]["hba1c_percent_baseline"] == pytest.approx([hb.value, hb.ci_low, hb.ci_high], abs=1e-12)
    for metric in ("balanced_accuracy", "macro_auroc"):
        assert supp["A"][metric]["cgm"] == pytest.approx(main2["C"][metric]["cgm"], abs=1e-12)
    assert supp["B"]["diagnostic stratum (4)"] == main2["D"]["diagnostic stratum (4)"]
