"""Figures: one folder per manuscript figure, holding only the manuscript figures; every plotted estimate equals its
source table; both versions of Figs 1 and 2 show the same values; labels and colours follow the conventions."""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cgmprofiles.config import get_config, results_dir

ROOT = Path(__file__).resolve().parents[1]
FIG = get_config().figures
AGG = results_dir("aggregate")
EXPECTED_FIGURES = {"figure_1": ["summary", "profiles"], "figure_2": ["summary", "correspondence"], "figure_3": ["summary", "summary_panels_a_b"]}


def figure_dir(figure: str) -> Path:
    return ROOT / "results" / "figures" / FIG[figure].folder


def plotted(figure: str, version: str) -> dict:
    path = figure_dir(figure) / "source_data" / f"{FIG[figure][version]}_plotted.json"
    if not path.exists():
        pytest.skip("figures not generated yet")
    return json.loads(path.read_text())


def test_results_figures_holds_one_folder_per_figure_and_nothing_else():
    figures = ROOT / "results" / "figures"
    if not figures.exists():
        pytest.skip("figures not generated yet")
    assert sorted(p.name for p in figures.iterdir()) == sorted(FIG[f].folder for f in EXPECTED_FIGURES)
    for figure, versions in EXPECTED_FIGURES.items():
        stems = {FIG[figure][v] for v in versions}
        images = {p.stem for p in figure_dir(figure).iterdir() if p.suffix in (".pdf", ".png")}
        assert images == stems, figure
        assert (figure_dir(figure) / f"make_figure_{figure.split('_')[1]}.py").exists()
        for stem in stems:
            for suffix in (".pdf", ".png"):
                assert (figure_dir(figure) / f"{stem}{suffix}").stat().st_size > 10_000


def test_figure_1_eta_and_counts_match_sources():
    summary, profiles = plotted("figure_1", "summary"), plotted("figure_1", "profiles")
    kw = pd.read_csv(AGG / "forward_kruskal_wallis.csv").query("canonical_stratum == 't2d_oral_non_insulin'").set_index(["window", "metric"])
    for key, panel in summary.items():
        window, metric = re.findall(r"'([^']+)'", key)
        assert panel["eta_squared"] == pytest.approx(kw.loc[(window, metric), "eta_squared"], abs=1e-12)
        assert [panel[c]["n"] for c in ("C1", "C2", "C3")] == [192, 111, 155]
        for c in ("C1", "C2", "C3"):
            assert profiles[key][c]["median"] == pytest.approx(panel[c]["median"], abs=1e-12)


def test_figure_2_panels_match_sources():
    values = plotted("figure_2", "summary")
    counts = pd.read_csv(AGG / "reverse_primary_contingency.csv")[["C1", "C2", "C3"]].to_numpy()
    assert np.array_equal(np.array(values["A"]["counts"]), counts)
    hba1c = pd.read_csv(AGG / "forward_vs_reverse_eta_squared.csv").query("hba1c_flag").iloc[0]
    assert values["B"]["HbA1c"] == pytest.approx([hba1c.eta_squared, hba1c.ci_low, hba1c.ci_high], abs=1e-12)
    b6 = pd.read_csv(AGG / "pooled_agreement.csv").query("metric == 'ARI' and target == 'diagnostic stratum (4)'").sort_values("pooled_k")
    for got, want in zip(values["D"]["diagnostic stratum (4)"], b6[["value", "ci_low", "ci_high"]].to_numpy().tolist()):
        assert got == pytest.approx(want, abs=1e-12)


def test_figure_2_correspondence_shows_the_same_values():
    summary, corr = plotted("figure_2", "summary"), plotted("figure_2", "correspondence")
    assert corr["A"]["counts"] == summary["A"]["counts"]
    assert corr["C"]["cgm"] == pytest.approx(summary["C"]["macro_auroc"]["cgm"], abs=1e-12)
    assert corr["C"]["both"] == pytest.approx(summary["C"]["macro_auroc"]["both"], abs=1e-12)
    assert corr["B"]["hba1c_percent_baseline"] == pytest.approx(summary["B"]["HbA1c"], abs=1e-12)
    r2 = pd.read_csv(AGG / "reverse_prediction.csv").query("direction == 'CGM to clinical variable' and feature_set == 'extended' and window == 'window_full'")
    hb = r2.set_index("target").loc["hba1c_percent_baseline"]
    assert corr["D"]["hba1c_percent_baseline"] == pytest.approx([hb.value, hb.ci_low, hb.ci_high], abs=1e-12)


def test_figure_3_specificity_matches_source():
    values = plotted("figure_3", "summary")["C"]
    spec = pd.read_csv(AGG / "strata_specificity.csv").query("variant == 'primary (T2D scaler, all)' and metric == 'coefficient_of_variation_pct'")
    for r in spec.itertuples():
        assert values[str((r.canonical_stratum, r.window))] == pytest.approx([r.mean_diff_c2_minus_rest, r.diff_ci_low, r.diff_ci_high], abs=1e-12)


def test_source_data_holds_no_participant_rows():
    for figure in EXPECTED_FIGURES:
        data = figure_dir(figure) / "source_data"
        if not data.exists():
            pytest.skip("figures not generated yet")
        for csv in data.glob("*.csv"):
            assert "participant_id" not in pd.read_csv(csv, nrows=1).columns, csv.name


def test_figure_3_without_panel_c_shows_the_same_values():
    full, a_b = plotted("figure_3", "summary"), plotted("figure_3", "summary_panels_a_b")
    assert "C" not in a_b and a_b["A"] == full["A"] and a_b["B"] == full["B"]


def test_summary_figures_use_the_record_palette():
    assert list(get_config().colors.summary_palette) == ["#BA2828", "#003366", "#5BBABA", "#FF0000", "#888888"]


def test_labels_and_colours_follow_conventions():
    source = "\n".join(p.read_text() for p in (ROOT / "src/cgmprofiles/plotting").glob("*.py"))
    assert "T2D oral" not in source and "#FF0000" not in source.upper()
    cfg = get_config().colors
    strata, profiles = set(cfg.strata.raw.values()), set(cfg.profiles.raw.values())
    assert len(strata) == 4 and not strata & profiles
