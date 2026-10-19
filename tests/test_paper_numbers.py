"""Every number quoted in the manuscript (docs/paper1_draft_2026-10-19.pdf), at the rounding used in the text,
against results/numbers/ and results/tables/. A manuscript value that does not match the outputs is a manuscript
error: it is listed in docs/MANUSCRIPT_FIXES.md and marked xfail here with the fix number."""
from __future__ import annotations

import pandas as pd
import pytest

from cgmprofiles.config import get_config, results_dir
from cgmprofiles.report import read_numbers
from cgmprofiles.tables import round_half_up

FULL, H48, CV = "window_full", "window_48h", "coefficient_of_variation_pct"


@pytest.fixture(scope="module")
def numbers() -> dict:
    try:
        return {k: read_numbers(k) for k in ("01_cohort", "02_forward", "03_reverse", "04_strata", "05_temporal")}
    except FileNotFoundError:
        pytest.skip("results/numbers not generated yet (bash run_all.sh)")


def r(value: float, decimals: int) -> str:
    return round_half_up(value, decimals)


def ci(values, decimals: int) -> str:
    v, low, high = values[:3]
    return f"{r(v, decimals)} [{r(low, decimals)}, {r(high, decimals)}]"


# (location in the manuscript, text in the manuscript, function of the numbers giving the same text)
CHECKS = [
    ("2.1 processed participants", "1,892", lambda n: f"{n['01_cohort']['n_parquet']:,}"),
    ("2.1 passed preprocessing", "1,591", lambda n: f"{n['01_cohort']['n_cohort']:,}"),
    ("2.1 strata", "564/405/458/164", lambda n: "/".join(str(n["01_cohort"]["stratum_n"][s]) for s in ("healthy", "pre_diabetes", "t2d_oral_non_insulin", "insulin_dependent"))),
    ("2.2 training and frozen assignment", "325/133", lambda n: f"{n['01_cohort']['t2d_train_n']}/{n['01_cohort']['t2d_assigned_frozen_n']}"),
    ("2.2 k-means initializations", "100", lambda n: str(n["01_cohort"]["kmeans_n_init"])),
    ("Table 1 n", "192/111/155", lambda n: "/".join(str(n["01_cohort"]["profile_counts"]["t2d_oral_non_insulin"][c]) for c in ("1", "2", "3"))),
    ("Table 1 footnote missing", "30/25/23/1", lambda n: "/".join(str(n["01_cohort"]["table1_missing"][k]) for k in ("HbA1c, %", "C-peptide, ng/mL", "Triglyceride-to-HDL ratio (mg/dL)", "Waist-to-hip ratio"))),
    ("3.1 GLP-1/GIP shares", "57/32/34", lambda n: "/".join(r(n["01_cohort"]["glp1_percent_by_profile"][c], 0) for c in ("C2", "C1", "C3"))),
    ("Abstract eta squared range", "0.042-0.078", lambda n: f"{r(min(n['02_forward']['eta_squared'].values()), 3)}-{r(max(n['02_forward']['eta_squared'].values()), 3)}"),
    ("3.1 full-record range", "0.052-0.078", lambda n: f"{r(min(v for k, v in n['02_forward']['eta_squared'].items() if k.startswith(FULL)), 3)}-"
                                                       f"{r(max(v for k, v in n['02_forward']['eta_squared'].items() if k.startswith(FULL)), 3)}"),
    ("3.1 all p < 0.0001", "True", lambda n: str(n["02_forward"]["max_p_t2d"] < 1e-4)),
    ("Table 2 mean glucose 48 h", "0.068 [0.029, 0.120]", lambda n: ci(n["02_forward"]["eta_squared_ci"][f"{H48}/mean_glucose"], 3)),
    ("Table 2 mean glucose full", "0.057 [0.021, 0.109]", lambda n: ci(n["02_forward"]["eta_squared_ci"][f"{FULL}/mean_glucose"], 3)),
    ("Table 2 TIR 48 h", "0.064 [0.025, 0.116]", lambda n: ci(n["02_forward"]["eta_squared_ci"][f"{H48}/time_in_range_pct"], 3)),
    ("Table 2 TIR full", "0.078 [0.038, 0.133]", lambda n: ci(n["02_forward"]["eta_squared_ci"][f"{FULL}/time_in_range_pct"], 3)),
    ("Table 2 CV 48 h", "0.042 [0.015, 0.087]", lambda n: ci(n["02_forward"]["eta_squared_ci"][f"{H48}/{CV}"], 3)),
    ("Table 2 CV full", "0.052 [0.023, 0.100]", lambda n: ci(n["02_forward"]["eta_squared_ci"][f"{FULL}/{CV}"], 3)),
    ("3.1 C1-C3 CV 48 h", "-0.76 [-2.06, 0.63]", lambda n: ci(n["02_forward"]["cv_c1_minus_c3"][H48], 2)),
    ("3.1 C1-C3 CV full", "-1.37 [-2.49, -0.21]", lambda n: ci(n["02_forward"]["cv_c1_minus_c3"][FULL], 2)),
    ("3.1 C2 lower than C1, 48 h", "2.26 [0.99, 3.56]", lambda n: ci(n["02_forward"]["cv_c1_minus_c2"][H48], 2)),
    ("3.1 C2 lower than C3, 48 h", "-3.02", lambda n: r(n["02_forward"]["cv_c2_minus_c3"][H48][0], 2)),
    ("3.1 CV eta training only, full", "0.021", lambda n: r(n["02_forward"]["cv_eta_training_only"][FULL], 3)),
    ("3.1 CV eta held-out, full", "0.139", lambda n: r(n["02_forward"]["cv_eta_heldout"][FULL], 3)),
    ("3.1 CV eta training only, 48 h", "0.013", lambda n: r(n["02_forward"]["cv_eta_training_only"][H48], 3)),
    ("3.1 CV eta held-out, 48 h", "0.111", lambda n: r(n["02_forward"]["cv_eta_heldout"][H48], 3)),
    ("3.1 split p full / 48 h", "0.043/0.084", lambda n: f"{r(n['02_forward']['split_dependence_p'][f'{FULL}/{CV}'], 3)}/{r(n['02_forward']['split_dependence_p'][f'{H48}/{CV}'], 3)}"),
    ("3.1 C2-C1 CV without GLP-1/GIP flag", "-2.77 [-4.46, -1.05]", lambda n: ci(n["02_forward"]["glp1_cv_c2_minus_c1"][f"{FULL}/no GLP-1/GIP flag"], 2)),
    ("3.1 C2-C1 CV with GLP-1/GIP flag", "0.51 [-0.99, 2.02]", lambda n: ci(n["02_forward"]["glp1_cv_c2_minus_c1"][f"{FULL}/GLP-1/GIP flag"], 2)),
    ("3.2 CGM group sizes", "188/182/88", lambda n: "/".join(str(n["03_reverse"]["group_sizes"][g]) for g in ("G1", "G2", "G3"))),
    ("3.2 median mean glucose", "120/147/185", lambda n: "/".join(r(n["03_reverse"]["median_mean_glucose"][g], 0) for g in ("G1", "G2", "G3"))),
    ("3.2 median CV", "16.4/21.1/24.8", lambda n: "/".join(r(n["03_reverse"]["median_cv"][g], 1) for g in ("G1", "G2", "G3"))),
    ("3.2 silhouette favoured two groups", "2", lambda n: str(int(n["03_reverse"]["silhouette"]["silhouette_selected_k"]))),
    ("Abstract and 3.2 ARI", "0.039 [0.020, 0.068]", lambda n: ci([n["03_reverse"]["ari"][k] for k in ("value", "ci_low", "ci_high")], 3)),
    ("3.2 ARI permutation p", "0.001", lambda n: r(n["03_reverse"]["ari"]["perm_p"], 3)),
    ("3.2 Cramer's V", "0.205", lambda n: r(n["03_reverse"]["cramers_v"], 3)),
    ("3.2 and Table 3 HbA1c", "0.422 [0.343, 0.497]", lambda n: ci(n["03_reverse"]["reverse_eta"]["hba1c_percent_baseline"], 3)),
    ("Abstract non-HbA1c range", "0.002-0.061", lambda n: "-".join(r(v, 3) for v in n["03_reverse"]["non_hba1c_eta_range"])),
    ("Table 3 age", "0.061 [0.024, 0.113]", lambda n: ci(n["03_reverse"]["reverse_eta"]["participants_age"], 3)),
    ("Table 3 BMI", "0.020 [0.001, 0.060]", lambda n: ci(n["03_reverse"]["reverse_eta"]["bmi_baseline"], 3)),
    ("Table 3 C-peptide", "0.002 [0.000, 0.029]", lambda n: ci(n["03_reverse"]["reverse_eta"]["c_peptide_ngml_baseline"], 3)),
    ("Table 3 TG/HDL", "0.025 [0.004, 0.069]", lambda n: ci(n["03_reverse"]["reverse_eta"]["tg_hdl_ratio"], 3)),
    ("Table 3 waist-to-hip", "0.028 [0.005, 0.069]", lambda n: ci(n["03_reverse"]["reverse_eta"]["waist_to_hip_ratio_baseline"], 3)),
    ("3.2 R2 HbA1c", "0.590 [0.455, 0.682]", lambda n: ci(n["03_reverse"]["cgm_to_clinical_r2"]["hba1c_percent_baseline"], 3)),
    ("Abstract R2 HbA1c", "0.59", lambda n: r(n["03_reverse"]["cgm_to_clinical_r2"]["hba1c_percent_baseline"][0], 2)),
    ("3.2 R2 age, BMI, waist-to-hip", "0.104/0.062/0.050", lambda n: "/".join(r(n["03_reverse"]["cgm_to_clinical_r2"][f][0], 3)
                                                                              for f in ("participants_age", "bmi_baseline", "waist_to_hip_ratio_baseline"))),
    ("3.2 R2 C-peptide, TG/HDL", "0.000/0.002", lambda n: "/".join(r(abs(n["03_reverse"]["cgm_to_clinical_r2"][f][0]), 3) for f in ("c_peptide_ngml_baseline", "tg_hdl_ratio"))),
    ("3.2 clinical to mean glucose and TIR", "0.51/0.51", lambda n: "/".join(r(n["03_reverse"]["clinical_to_cgm_r2"][f"{FULL}/{m}"][0], 2) for m in ("mean_glucose", "time_in_range_pct"))),
    ("3.2 without HbA1c", "0.02/0.04", lambda n: "/".join(r(n["03_reverse"]["clinical_to_cgm_r2"][f"{FULL}/{m}"][1], 2) for m in ("mean_glucose", "time_in_range_pct"))),
    ("3.2 CV with and without HbA1c", "0.079/0.051", lambda n: "/".join(r(v, 3) for v in n["03_reverse"]["clinical_to_cgm_r2"][f"{FULL}/{CV}"])),
    ("Abstract and 3.2 AUROC", "0.649 [0.612, 0.686]", lambda n: ci(n["03_reverse"]["profile_auroc"], 3)),
    ("3.2 balanced accuracy", "0.448 [0.400, 0.496]", lambda n: ci(n["03_reverse"]["profile_balanced_accuracy"], 3)),
    ("3.2 chance balanced accuracy", "0.333", lambda n: r(n["03_reverse"]["chance_balanced_accuracy"], 3)),
    ("3.2 permutation p (both metrics)", "0.005/0.005", lambda n: f"{r(n['03_reverse']['profile_auroc'][3], 3)}/{r(n['03_reverse']['profile_balanced_accuracy'][3], 3)}"),
    ("Abstract and 3.2 AUROC HbA1c to combined", "0.605/0.660", lambda n: f"{r(n['03_reverse']['hba1c_alone_auroc'], 3)}/{r(n['03_reverse']['hba1c_plus_cgm_auroc'], 3)}"),
    ("3.2 AUROC gain", "0.055 [0.017, 0.094]", lambda n: ci(n["03_reverse"]["auroc_gain"], 3)),
    ("3.2 3 of 15 combinations", "3 of 15", lambda n: f"{n['03_reverse']['age_cells_at_or_above_medium']} of {n['03_reverse']['age_cells_total']}"),
    ("3.3 healthy mean glucose", "0.021 [0.003, 0.054]", lambda n: ci(n["04_strata"]["forward_mean_glucose_full"]["healthy"], 3)),
    ("3.3 prediabetes mean glucose", "0.071 [0.028, 0.124]", lambda n: ci(n["04_strata"]["forward_mean_glucose_full"]["pre_diabetes"], 3)),
    ("3.3 CV negligible", "<= 0.002", lambda n: "<= 0.002" if float(r(n["04_strata"]["forward_cv_max_healthy_prediabetes"], 3)) <= 0.002 else "no"),
    ("3.3 assigned to C2", "88/89", lambda n: f"{n['04_strata']['assigned_c2']['healthy']}/{n['04_strata']['assigned_c2']['pre_diabetes']}"),
    ("3.3 healthy C2 minus rest", "0.17 [-0.77, 1.12]", lambda n: ci(n["04_strata"]["cv_c2_minus_rest"][f"healthy/{FULL}"], 2)),
    ("3.3 prediabetes C2 minus rest", "-1.04 [-1.88, -0.12]", lambda n: ci(n["04_strata"]["cv_c2_minus_rest"][f"pre_diabetes/{FULL}"], 2)),
    ("3.3 T2D C2 minus rest", "-2.39 [-3.41, -1.43]", lambda n: ci(n["04_strata"]["cv_c2_minus_rest_t2d"][FULL], 2)),
    ("3.3 Holm p prediabetes", "0.065", lambda n: r(n["04_strata"]["cv_c2_minus_rest"][f"pre_diabetes/{FULL}"][4], 3)),
    ("Abstract prediabetes share (approximately 40%)", "37%-44%", lambda n: f"{r(n['04_strata']['prediabetes_share_of_t2d'][H48] * 100, 0)}%-{r(n['04_strata']['prediabetes_share_of_t2d'][FULL] * 100, 0)}%"),
    ("3.3 ARI other strata", "0.004-0.013", lambda n: f"{r(min(n['04_strata']['ari_other_strata'][s] for s in ('healthy', 'pre_diabetes', 'insulin_dependent')), 3)}-"
                                                      f"{r(max(n['04_strata']['ari_other_strata'][s] for s in ('healthy', 'pre_diabetes', 'insulin_dependent')), 3)}"),
    ("Abstract ARI healthy and prediabetes", "<= 0.013", lambda n: "<= 0.013" if max(float(r(n["04_strata"]["ari_other_strata"][s], 3)) for s in ("healthy", "pre_diabetes")) <= 0.013 else "no"),
    ("3.3 AUROC healthy, prediabetes", "0.685/0.741", lambda n: "/".join(r(n["04_strata"]["auroc"][s][0], 3) for s in ("healthy", "pre_diabetes"))),
    ("3.3 AUROC HbA1c alone", "0.664/0.628", lambda n: "/".join(r(n["04_strata"]["auroc"][s][1], 3) for s in ("healthy", "pre_diabetes"))),
    ("3.3 insulin-treated C-peptide full", "0.085 [0.019, 0.197]", lambda n: ci(n["04_strata"]["insulin_c_peptide"][FULL], 3)),
    ("3.3 insulin-treated C-peptide 48 h", "0.021", lambda n: r(n["04_strata"]["insulin_c_peptide"][H48][0], 3)),
    ("Abstract and 3.3 pooled ARI with stratum", "0.160 [0.136, 0.187]", lambda n: ci(n["04_strata"]["pooled_ari_diagnosis_k3"], 3)),
    ("3.3 pooled ARI p", "0.001", lambda n: r(n["04_strata"]["pooled_ari_diagnosis_k3"][3], 3)),
    ("3.3 pooled versus separate T2D clustering", "0.037/0.039", lambda n: f"{r(n['04_strata']['pooled_ari_t2d_profiles_k3'], 3)}/{r(n['04_strata']['t2d_only_ari'], 3)}"),
    ("3.3 combined labels and null", "0.093/0.088/0.001", lambda n: f"{r(n['04_strata']['pooled_ari_combined_k3'], 3)}/{r(n['04_strata']['pooled_within_null'][0], 3)}/"
                                                                    f"{r(n['04_strata']['pooled_within_null'][1], 3)}"),
    ("Abstract windows ARI", "0.42", lambda n: r(n["05_temporal"]["heldout_ari"][0], 2)),
    ("Abstract profile ARI per window", "0.030/0.036", lambda n: f"{r(n['05_temporal']['profile_ari_window_a'], 3)}/{r(n['05_temporal']['profile_ari_window_b'], 3)}"),
]


# Manuscript values that do not match the outputs (rounded twice by hand); see docs/MANUSCRIPT_FIXES.md
KNOWN_ERRORS = {
    "Table 3 waist-to-hip": "MANUSCRIPT_FIXES.md item 8: CI lower bound 0.004454 rounds to 0.004, not 0.005",
    "3.3 healthy mean glucose": "MANUSCRIPT_FIXES.md item 9: CI upper bound 0.053478 rounds to 0.053, not 0.054",
    "3.3 pooled versus separate T2D clustering": "MANUSCRIPT_FIXES.md item 10: pooled ARI 0.036452 rounds to 0.036, not 0.037",
}


@pytest.mark.parametrize("where,manuscript,compute", CHECKS, ids=[c[0] for c in CHECKS])
def test_manuscript_number(numbers, where, manuscript, compute):
    if where in KNOWN_ERRORS:
        assert compute(numbers).replace("\u2212", "-") != manuscript
        pytest.xfail(KNOWN_ERRORS[where])
    assert compute(numbers).replace("−", "-") == manuscript


def test_methods_constants_match_config():
    cfg = get_config()
    assert (cfg.bootstrap.n_default, cfg.permutation.n_agreement, cfg.permutation.n_prediction, cfg.permutation.n_prediction_escalated) == (2000, 1000, 200, 1000)
    assert (cfg.prediction.n_folds, cfg.prediction.n_repeats, cfg.profiles.kmeans_n_init, tuple(cfg.cgm.tir_mgdl)) == (5, 20, 100, (70.0, 180.0))


# Table 1 as printed in the manuscript (C1, C2, C3, All, P)
TABLE1 = {
    "n": ["192", "111", "155", "458", ""],
    "Age, years": ["64 [56–70]", "51 [47–58]", "68 [63–76]", "63 [54–70]", "<0.001"],
    "Female sex": ["114 (59.4)", "85 (76.6)", "48 (31.0)", "247 (53.9)", "<0.001"],
    "Race": ["", "", "", "", "<0.001"],
    "    White": ["80 (41.7)", "43 (38.7)", "90 (58.1)", "213 (46.5)", ""],
    "    Black or African American": ["55 (28.6)", "45 (40.5)", "22 (14.2)", "122 (26.6)", ""],
    "    Asian": ["52 (27.1)", "11 (9.9)", "29 (18.7)", "92 (20.1)", ""],
    "    Other or not reported": ["5 (2.6)", "12 (10.8)", "14 (9.0)", "31 (6.8)", ""],
    "BMI, kg/m²": ["26.8 [24.0–29.7]", "38.7 [35.7–42.8]", "29.1 [26.2–31.9]", "29.6 [25.9–35.0]", "<0.001"],
    "HbA1c, %": ["6.2 [5.7–6.6]", "6.2 [5.8–6.7]", "6.7 [6.2–7.3]", "6.3 [5.8–6.9]", "<0.001"],
    "C-peptide, ng/mL": ["2.3 [1.7–3.1]", "3.3 [2.4–4.9]", "4.4 [3.2–6.0]", "3.1 [2.2–4.4]", "<0.001"],
    "Triglyceride-to-HDL ratio": ["1.8 [1.3–2.5]", "2.8 [1.9–3.9]", "4.5 [3.1–6.7]", "2.7 [1.7–4.2]", "<0.001"],
    "Waist-to-hip ratio": ["0.91 [0.86–0.96]", "0.91 [0.87–0.97]", "1.00 [0.95–1.05]", "0.94 [0.89–0.99]", "<0.001"],
    "Metformin": ["140 (72.9)", "62 (55.9)", "109 (70.3)", "311 (67.9)", "0.007"],
    "GLP-1 or GIP/GLP-1 agonist": ["62 (32.3)", "63 (56.8)", "53 (34.2)", "178 (38.9)", "<0.001"],
    "SGLT2 inhibitor": ["33 (17.2)", "16 (14.4)", "50 (32.3)", "99 (21.6)", "<0.001"],
    "Sulfonylurea": ["13 (6.8)", "5 (4.5)", "30 (19.4)", "48 (10.5)", "<0.001"],
}
KNOWN_TABLE1_ERRORS = {"C-peptide, ng/mL": "MANUSCRIPT_FIXES.md item 1: C1 lower quartile 1.6475 rounds to 1.6, not 1.7"}


@pytest.mark.parametrize("row", list(TABLE1))
def test_table1_row(row):
    path = results_dir("tables") / "table1_baseline.csv"
    if not path.exists():
        pytest.skip("Table 1 not generated yet")
    table = pd.read_csv(path, dtype=str, keep_default_na=False).set_index("characteristic")
    if row in KNOWN_TABLE1_ERRORS:
        pytest.xfail(KNOWN_TABLE1_ERRORS[row])
    assert table.loc[row, ["C1", "C2", "C3", "All", "p_value"]].tolist() == TABLE1[row]


TABLE2 = [["Mean glucose", "0.068 [0.029, 0.120]", "0.057 [0.021, 0.109]", "Higher in C3 than C1 and C2"],
          ["Time in range", "0.064 [0.025, 0.116]", "0.078 [0.038, 0.133]", "Lower in C3 than C1 and C2"],
          ["Coefficient of variation", "0.042 [0.015, 0.087]", "0.052 [0.023, 0.100]", "Lower in C2 than C1 and C3"]]
TABLE3 = [["Forward", "Mean glucose", "0.057 [0.021, 0.109]"], ["", "Time in range", "0.078 [0.038, 0.133]"], ["", "Coefficient of variation", "0.052 [0.023, 0.100]"],
          ["Reverse", "Age", "0.061 [0.024, 0.113]"], ["", "BMI", "0.020 [0.001, 0.060]"], ["", "C-peptide", "0.002 [0.000, 0.029]"],
          ["", "Triglyceride-to-HDL ratio", "0.025 [0.004, 0.069]"], ["", "Waist-to-hip ratio", "0.028 [0.005, 0.069]"], ["", "HbA1c", "0.422 [0.343, 0.497]"]]


@pytest.mark.parametrize("name,expected", [("table2_forward", TABLE2), ("table3_bidirectional", TABLE3)])
def test_tables_2_and_3(name, expected):
    path = results_dir("tables") / f"{name}.csv"
    if not path.exists():
        pytest.skip(f"{name} not generated yet")
    table = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = [[c.replace("−", "-") for c in row] for row in table.values.tolist()]
    if name == "table3_bidirectional":  # manuscript waist-to-hip CI lower bound 0.005 is a rounding error (MANUSCRIPT_FIXES.md item 8)
        assert rows[7] == ["", "Waist-to-hip ratio", "0.028 [0.004, 0.069]"] and expected[7][2] == "0.028 [0.005, 0.069]"
        rows, expected = rows[:7] + rows[8:], expected[:7] + expected[8:]
    assert rows == expected
