"""Paper tables: Table 1 and Supplementary Tables S1 and S2 (baseline), Table 2 (forward) and
Table 3 (both directions).

Baseline tables: observed values only; laboratory values dated after the end of CGM are missing.
Continuous: median [IQR], Kruskal-Wallis. Categorical: n (%), chi-square, or Fisher's exact test
when an expected count is below 5 (Fisher-Freeman-Halton by Monte Carlo for larger tables). P values
come from unmasked counts and are descriptive. Small cells are masked block by block
(disclosure.mask_table_block); race categories with a cell of 1 to 4 are combined first.
"""
from __future__ import annotations

from decimal import ROUND_HALF_UP, Decimal
from math import lgamma

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, fisher_exact, kruskal

from .config import get_config
from .disclosure import MASK_COMPLEMENT, MASK_SMALL, SMALL_MAX, SMALL_MIN, mask_table_block

CFG = get_config()
TCFG = CFG.disclosure
COLLAPSED_RACE = "Other, multiple or not reported"
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
EN_DASH = "–"
MINUS = "−"


# ---------------------------------------------------------------- number formatting
def round_half_up(value: float, decimals: int) -> str:
    """Decimal half-up rounding of a float, after removing binary noise (6.1499999999 -> 6.15 -> 6.2)."""
    exact = Decimal(repr(round(float(value), 10)))
    return str(exact.quantize(Decimal(1).scaleb(-decimals), rounding=ROUND_HALF_UP))


def fmt_signed(value: float, decimals: int) -> str:
    """Half-up rounding with a typographic minus sign."""
    return round_half_up(value, decimals).replace("-", MINUS)


def fmt_ci(value: float, low: float, high: float, decimals: int = 3) -> str:
    """'0.057 [0.021, 0.109]'."""
    return f"{fmt_signed(value, decimals)} [{fmt_signed(low, decimals)}, {fmt_signed(high, decimals)}]"


def fmt_p_manuscript(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return f"<{TCFG.p_floor}" if p < TCFG.p_floor else round_half_up(p, 3)


def fmt_p_record(p: float) -> str:
    if p is None or not np.isfinite(p):
        return ""
    return f"<{TCFG.p_floor}" if p < TCFG.p_floor else f"{float(f'{p:.2g}'):g}"


# ---------------------------------------------------------------- tests
def _log_table_probability(table: np.ndarray) -> float:
    rows, cols, n = table.sum(axis=1), table.sum(axis=0), table.sum()
    return (sum(lgamma(r + 1) for r in rows) + sum(lgamma(c + 1) for c in cols) - lgamma(n + 1) - sum(lgamma(x + 1) for x in table.ravel()))


def categorical_p(table: np.ndarray) -> tuple[float, str]:
    """Chi-square, or Fisher (2 x 2) / Fisher-Freeman-Halton Monte Carlo when an expected count is below 5."""
    table = table[table.sum(axis=1) > 0][:, table.sum(axis=0) > 0]
    if table.shape[0] < 2 or table.shape[1] < 2:
        return np.nan, ""
    _, p, _, expected = chi2_contingency(table, correction=False)
    if expected.min() >= TCFG.fisher_expected_min:
        return float(p), "chi-square"
    if table.shape == (2, 2):
        return float(fisher_exact(table)[1]), "Fisher exact"
    observed = _log_table_probability(table)
    rng = np.random.default_rng(CFG.seeds.table_monte_carlo)
    rows = np.repeat(np.arange(table.shape[0]), table.sum(axis=1))
    cols = np.repeat(np.arange(table.shape[1]), table.sum(axis=0))
    hits = 0
    for _ in range(TCFG.monte_carlo_n):
        sim = np.zeros_like(table)
        np.add.at(sim, (rows, rng.permutation(cols)), 1)
        hits += _log_table_probability(sim) <= observed + 1e-9
    return float((hits + 1) / (TCFG.monte_carlo_n + 1)), "Fisher-Freeman-Halton (Monte Carlo)"


# ---------------------------------------------------------------- baseline blocks
def build_baseline_blocks(sub: pd.DataFrame, sections: list) -> tuple[list[str], list[dict]]:
    """Unformatted blocks (quartiles, counts, p) for one stratum; ``sections`` = [(section, [(kind, column, label)])]."""
    profiles = sorted(sub.label.unique())
    cols = [f"C{p}" for p in profiles] + ["All"]
    groups = {f"C{p}": sub[sub.label == p] for p in profiles} | {"All": sub}
    blocks = [{"section": "", "label": "n", "kind": "n", "values": {c: len(g) for c, g in groups.items()}}]
    race = CFG.columns.race
    for section, items in sections:
        for kind, column, label in items:
            if kind == "text":
                blocks.append({"section": section, "label": label, "kind": "text", "values": {c: "not recorded" for c in cols}})
                continue
            if kind == "cont":
                present = [sub.loc[sub.label == p, column].dropna() for p in profiles]
                quartiles = {c: g[column].dropna().quantile([0.25, 0.5, 0.75]).to_numpy() for c, g in groups.items()}
                scale = {c: float(g[column].dropna().abs().median()) for c, g in groups.items()}
                blocks.append({"section": section, "label": label, "kind": "cont", "column": column, "quartiles": quartiles, "scale": scale,
                               "missing": int(sub[column].isna().sum()), "p": float(kruskal(*present).pvalue), "test": "Kruskal-Wallis"})
                continue
            if kind == "bin":
                cats, filled = [1], sub[column]
                denominators = {c: int(g[column].notna().sum()) for c, g in groups.items()}
            else:
                filled = sub[column].fillna("Missing")
                cats = list(CFG.cohort.split_order) if column == "split" else filled.value_counts().index.tolist()
                if column == race:
                    small = [cat for cat in cats if any(SMALL_MIN <= int(((filled == cat) & (sub.label == p)).sum()) <= SMALL_MAX for p in profiles)
                             or SMALL_MIN <= int((filled == cat).sum()) <= SMALL_MAX]
                    if small:
                        filled = filled.where(~filled.isin(small), COLLAPSED_RACE)
                        cats = [c for c in cats if c not in small] + [COLLAPSED_RACE]
                denominators = {c: len(g) for c, g in groups.items()}
            counts = np.array([[int(((sub.label == p) & (filled == cat)).sum()) for p in profiles] + [int((filled == cat).sum())] for cat in cats])
            table = (np.array([[int(((sub.label == p) & (sub[column] == v)).sum()) for p in profiles] for v in (1, 0)])
                     if kind == "bin" else counts[:, :-1])
            p, test = categorical_p(table)
            blocks.append({"section": section, "label": label, "kind": kind, "column": column, "categories": cats, "counts": counts,
                           "denominators": denominators, "missing": int(sub[column].isna().sum()), "p": p, "test": test, "exhaustive": kind == "cat"})
    for block in blocks:
        if block["kind"] in ("bin", "cat"):
            block["small"], block["complement"], block["ranges"] = mask_table_block(block, cols)
    return cols, blocks


def _cell(block: dict, r: int, c: int, cols: list[str]) -> str:
    if block["small"][r, c]:
        return MASK_SMALL
    if block["complement"][r, c]:
        return MASK_COMPLEMENT
    n = block["counts"][r, c]
    return f"{n} ({n / block['denominators'][cols[c]] * 100:.1f})"


def render_baseline(cols: list[str], blocks: list[dict], style: str, decimals: dict | None = None,
                    labels: dict | None = None, keep: list[str] | None = None) -> pd.DataFrame:
    """Masked table. style='record' reproduces the record's CSV cell for cell; style='paper' uses the
    manuscript conventions (half-up rounding to ``decimals``, en-dash IQR, P to 3 decimals)."""
    labels = labels or {}
    rows = []
    for b in blocks:
        if keep is not None and b["label"] not in keep:
            continue
        name = labels.get(b["label"], b["label"])
        p_text = (fmt_p_record if style == "record" else fmt_p_manuscript)(b.get("p", np.nan))
        base = {"section": b["section"], "characteristic": name}
        tail = {"missing_n": str(b.get("missing", "")), "p_value": p_text, "test": b.get("test", "")}
        if b["kind"] in ("n", "text"):
            rows.append({**base, **{c: str(v) for c, v in b["values"].items()}, **tail})
        elif b["kind"] == "cont":
            values = {}
            for c in cols:
                q1, med, q3 = b["quartiles"][c]
                if style == "record":
                    s = b["scale"][c]
                    d = 2 if s < 10 else (1 if s < 1000 else 0)
                    values[c] = f"{med:.{d}f} [{q1:.{d}f}, {q3:.{d}f}]"
                else:
                    d = decimals[b["column"]]
                    values[c] = f"{round_half_up(med, d)} [{round_half_up(q1, d)}{EN_DASH}{round_half_up(q3, d)}]"
            rows.append({**base, **values, **tail})
        elif b["kind"] == "bin":
            rows.append({**base, **{c: _cell(b, 0, j, cols) for j, c in enumerate(cols)}, **tail})
        else:
            rows.append({**base, **{c: "" for c in cols}, **tail})
            for r, cat in enumerate(b["categories"]):
                cat_name = labels.get(cat, SPLIT_LABELS.get(cat, cat))
                rows.append({"section": b["section"], "characteristic": f"    {cat_name}", **{c: _cell(b, r, j, cols) for j, c in enumerate(cols)},
                             "missing_n": "", "p_value": "", "test": ""})
    return pd.DataFrame(rows)


def masking_report(blocks: list[dict], cols: list[str], stratum: str) -> list[dict]:
    """Every masked cell with its feasible range (must not be a single value)."""
    out = []
    for b in blocks:
        for cell, (lo, hi) in b.get("ranges", {}).items():
            out.append({"stratum": stratum, "row": b["label"], "category": str(b["categories"][cell[0]]), "column": cols[cell[1]],
                        "label": MASK_SMALL if b["small"][cell] else MASK_COMPLEMENT, "recoverable": lo == hi})
    return out


# ---------------------------------------------------------------- Tables 2 and 3
def between_profile_pattern(pairs: pd.DataFrame, metric: str, windows: list[str], names: dict[int, str]) -> str:
    """'Higher in C3 than C1 and C2' when every pairwise CI involving one profile excludes 0 in the same
    direction in every window (Section 3.1 bootstrap differences)."""
    rows = pairs[pairs.metric == metric]
    for focal in sorted(names):
        signs = []
        for _, r in rows[(rows.cluster_a == focal) | (rows.cluster_b == focal)].iterrows():
            if r.window not in windows or not r.ci_excludes_zero:
                break
            diff = r.mean_diff_a_minus_b if r.cluster_a == focal else -r.mean_diff_a_minus_b
            signs.append(np.sign(diff))
        else:
            if signs and len(set(signs)) == 1 and len(signs) == 2 * len(windows):
                others = " and ".join(names[c] for c in sorted(names) if c != focal)
                return f"{'Higher' if signs[0] > 0 else 'Lower'} in {names[focal]} than {others}"
    return "No consistent single-profile difference"


def latex_escape(text: str) -> str:
    for k, v in {"%": r"\%", "&": r"\&", "_": r"\_", "#": r"\#", "²": r"$^2$", "<": r"$<$", MINUS: r"$-$", EN_DASH: "--"}.items():
        text = text.replace(k, v)
    return text


def simple_latex(table: pd.DataFrame, caption: str, label: str, footnote: str = "", column_spec: str | None = None) -> str:
    """booktabs table environment from a DataFrame of strings."""
    spec = column_spec or "l" * len(table.columns)
    lines = [r"\begin{table}[htbp]", r"\centering", r"\footnotesize", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{tabular}}{{{spec}}}", r"\toprule", " & ".join(latex_escape(str(c)) for c in table.columns) + r" \\", r"\midrule"]
    for row in table.itertuples(index=False):
        cells = [latex_escape(str(v)) for v in row]
        cells[0] = cells[0].replace("    ", r"\hspace{1em}")
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    if footnote:
        lines += [r"\par\smallskip", r"\begin{minipage}{\linewidth}\scriptsize", latex_escape(footnote), r"\end{minipage}"]
    return "\n".join(lines + [r"\end{table}", ""])
