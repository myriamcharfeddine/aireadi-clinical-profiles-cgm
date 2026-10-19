"""Small-cell disclosure control (docs/small_cell_policy.md).

Counts of 1 to 4 are shown as "<5"; while any masked value could be worked out from what is
shown, one more cell is masked as "NR". Recovery is checked with two integer programs per
masked value (its smallest and largest value consistent with everything shown).

Two solvers, each ported unchanged from the record so the same cells are masked:
- ``mask_table_block``: one block of a descriptive table (rows sum to All; categories of an
  exhaustive block sum to each column's denominator);
- ``MaskProblem``: named cells linked by arbitrary linear equalities (tables that share margins).
"""
from __future__ import annotations

from collections.abc import Hashable

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp

from .config import get_config

CFG = get_config().disclosure
SMALL_MIN, SMALL_MAX = CFG.small_min, CFG.small_max
MASK_SMALL, MASK_COMPLEMENT = CFG.mask_small, CFG.mask_complement
HIDDEN = "hidden"


def is_small(value: int) -> bool:
    """True for counts of 1 to 4."""
    return SMALL_MIN <= value <= SMALL_MAX


# ---------------------------------------------------------------- descriptive table blocks
def block_recovery_ranges(block: dict, mask: np.ndarray, cols: list[str]) -> dict[tuple[int, int], tuple[int, int]]:
    """Min and max of every masked cell of a block over integer tables consistent with what is shown."""
    counts = block["counts"]
    cells = [tuple(ix) for ix in np.argwhere(mask)]
    if not cells:
        return {}
    index = {cell: i for i, cell in enumerate(cells)}
    n_profiles = counts.shape[1] - 1
    lower, upper = [], []
    for (r, c) in cells:
        small = is_small(counts[r, c])
        lower.append(SMALL_MIN if small else SMALL_MAX + 1)
        upper.append(SMALL_MAX if small else block["denominators"][cols[c]])
    rows_a, rows_b = [], []

    def add_equation(members, total_known):
        coef, known = np.zeros(len(cells)), 0
        for cell, sign in members:
            if cell in index:
                coef[index[cell]] += sign
            else:
                known += sign * counts[cell]
        if coef.any():
            rows_a.append(coef)
            rows_b.append(total_known - known)

    for r in range(counts.shape[0]):
        add_equation([((r, c), 1) for c in range(n_profiles)] + [((r, n_profiles), -1)], 0)
    if block["exhaustive"]:
        for c in range(counts.shape[1]):
            add_equation([((r, c), 1) for r in range(counts.shape[0])], block["denominators"][cols[c]])
    constraints = [LinearConstraint(np.array(rows_a), rows_b, rows_b)] if rows_a else []
    bounds, integrality = Bounds(lower, upper), np.ones(len(cells))
    ranges = {}
    for cell, i in index.items():
        objective = np.zeros(len(cells))
        objective[i] = 1
        low = milp(objective, constraints=constraints, bounds=bounds, integrality=integrality)
        high = milp(-objective, constraints=constraints, bounds=bounds, integrality=integrality)
        if not (low.success and high.success):
            raise RuntimeError(f"Recovery check infeasible for {block['label']}")
        ranges[cell] = (int(round(low.fun)), int(round(-high.fun)))
    return ranges


def mask_table_block(block: dict, cols: list[str]) -> tuple[np.ndarray, np.ndarray, dict]:
    """Small-cell and complementary masks for one block, and the final recovery ranges."""
    counts = block["counts"]
    small = (counts >= SMALL_MIN) & (counts <= SMALL_MAX)
    complement = np.zeros_like(small)
    while True:
        ranges = block_recovery_ranges(block, small | complement, cols)
        exposed = [cell for cell, (lo, hi) in ranges.items() if lo == hi]
        if not exposed:
            return small, complement, ranges
        r, c = exposed[0]
        visible = ~(small | complement) & (counts > 0)
        candidates = [(counts[r, cc], (r, cc)) for cc in range(counts.shape[1] - 1) if visible[r, cc]]
        if not candidates and visible[r, -1]:
            candidates = [(counts[r, -1], (r, counts.shape[1] - 1))]
        if not candidates and block["exhaustive"]:
            candidates = [(counts[rr, c], (rr, c)) for rr in range(counts.shape[0]) if visible[rr, c]]
        if not candidates:
            raise RuntimeError(f"Cannot protect {block['label']}: no cell left to mask")
        complement[min(candidates)[1]] = True


# ---------------------------------------------------------------- linked cells
class MaskProblem:
    """Named integer cells linked by linear equalities; ``solve`` adds complementary masks."""

    def __init__(self, upper_bound: int):
        self.values: dict[Hashable, int] = {}
        self.state: dict[Hashable, str] = {}
        self.constraints: list[dict[Hashable, int]] = []
        self.protected: set = set()
        self.sensitive: set = set()
        self.fixed: set = set()
        self.upper_bound = upper_bound

    def add(self, key, value: int, hidden: bool = False, protected: bool = False, sensitive: bool = False, fixed: bool = False) -> None:
        """protected: masked only if nothing else works; fixed: never masked (already public)."""
        self.values[key] = int(value)
        if fixed:
            self.fixed.add(key)
        self.state[key] = HIDDEN if hidden else (MASK_SMALL if is_small(value) else "")
        if protected:
            self.protected.add(key)
        if sensitive:
            self.sensitive.add(key)

    def equal(self, total, parts: list) -> None:
        """Constraint: sum(parts) = total."""
        row = {total: -1}
        for key in parts:
            row[key] = row.get(key, 0) + 1
        self.constraints.append(row)

    def ranges(self, keys: list | None = None) -> dict:
        """Smallest and largest consistent value of every unknown (masked or hidden) cell."""
        unknown = [k for k, s in self.state.items() if s]
        if not unknown:
            return {}
        position = {k: i for i, k in enumerate(unknown)}
        a_rows, rhs = [], []
        for row in self.constraints:
            a, known = np.zeros(len(unknown)), 0
            for key, coeff in row.items():
                if key in position:
                    a[position[key]] += coeff
                else:
                    known -= coeff * self.values[key]
            if a.any():
                a_rows.append(a), rhs.append(known)
        lower = [SMALL_MIN if self.state[k] == MASK_SMALL else SMALL_MAX + 1 if self.state[k] == MASK_COMPLEMENT else 0 for k in unknown]
        upper = [SMALL_MAX if self.state[k] == MASK_SMALL else self.upper_bound for k in unknown]
        kwargs = {"bounds": Bounds(lower, upper), "integrality": np.ones(len(unknown))}
        if a_rows:
            kwargs["constraints"] = LinearConstraint(np.array(a_rows), rhs, rhs)
        out = {}
        for key, i in position.items():
            if keys is not None and key not in keys:
                continue
            objective = np.zeros(len(unknown))
            objective[i] = 1
            low, high = milp(objective, **kwargs), milp(-objective, **kwargs)
            out[key] = (int(round(low.fun)), int(round(-high.fun)))
        return out

    def must_protect(self, key) -> bool:
        state = self.state[key]
        return state == MASK_SMALL or (state == HIDDEN and is_small(self.values[key])) or key in self.sensitive

    def _recoverable(self, ranges: dict) -> list:
        return [k for k, (low, high) in ranges.items() if low == high and self.must_protect(k)]

    def solve(self) -> dict:
        """Greedy: while a protected quantity is recoverable, mask the nearby shown cell that leaves
        the fewest recoverable quantities (ties: unprotected, then smaller cells). An NR cell that is
        still determined by what is shown is shown again, since it hides nothing."""
        recoverable = self._recoverable(self.ranges())
        while recoverable:
            maskable = lambda keys: [k for k in keys if self.state[k] == "" and self.values[k] > SMALL_MAX and k not in self.fixed]  # noqa: E731
            linked, candidates = set(recoverable), []
            for _ in range(CFG.max_hops):
                for row in [r for r in self.constraints if any(k in r for k in linked)]:
                    linked.update(row)
                candidates = maskable(linked)
                if candidates:
                    break
            if not candidates:
                candidates = maskable(self.state)
            if not candidates:
                raise RuntimeError(f"Cannot protect {recoverable}")
            scored = []
            for key in candidates:
                self.state[key] = MASK_COMPLEMENT
                still = self._recoverable(self.ranges(keys=recoverable))
                scored.append((len(still), key in self.protected, self.values[key], str(key), key, still))
                self.state[key] = ""
            *_, best, still = min(scored, key=lambda t: t[:4])
            self.state[best] = MASK_COMPLEMENT
            recoverable = still
        for key, (low, high) in self.ranges().items():
            if self.state[key] == MASK_COMPLEMENT and low == high:
                self.state[key] = ""
        self.final_ranges = self.ranges()
        return self.final_ranges

    def shown(self, key) -> str:
        """Display text of a cell: its count, "<5" or "NR"."""
        state = self.state[key]
        if state == HIDDEN:
            raise KeyError(f"{key} is hidden")
        return state if state else str(self.values[key])

    def is_masked(self, key) -> bool:
        return self.state[key] != ""

    def check(self) -> dict:
        """Counts of masked cells and of protected quantities that remain recoverable (must be 0)."""
        ranges = getattr(self, "final_ranges", None) or self.ranges()
        protect = [k for k in ranges if self.must_protect(k)]
        return {"n_small_masked": sum(s == MASK_SMALL for s in self.state.values()),
                "n_complement_masked": sum(s == MASK_COMPLEMENT for s in self.state.values()),
                "n_protected_quantities": len(protect), "n_recoverable": sum(ranges[k][0] == ranges[k][1] for k in protect)}


def masked_count(value: int) -> str:
    """A single count as it may be shown: "<5" for 1 to 4."""
    return MASK_SMALL if is_small(value) else str(int(value))
