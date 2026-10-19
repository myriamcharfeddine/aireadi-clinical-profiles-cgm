"""Small-cell masking and the recovery check on toy tables."""
import numpy as np

from cgmprofiles.disclosure import MASK_COMPLEMENT, MASK_SMALL, MaskProblem, mask_table_block


def block(counts, exhaustive=True):
    counts = np.array(counts)
    cols = ["C1", "C2", "All"]
    denominators = {c: int(counts[:, j].sum()) for j, c in enumerate(cols)}
    return {"label": "toy", "counts": counts, "denominators": denominators, "exhaustive": exhaustive}, cols


def test_small_cell_masked_with_complement_and_unrecoverable():
    b, cols = block([[3, 40, 43]], exhaustive=False)
    b["denominators"] = {"C1": 53, "C2": 100, "All": 153}
    small, complement, ranges = mask_table_block(b, cols)
    assert small[0, 0] and complement[0, 1]
    assert all(lo < hi for lo, hi in ranges.values())


def test_no_masking_without_small_cells():
    b, cols = block([[30, 40, 70], [50, 60, 110]])
    small, complement, ranges = mask_table_block(b, cols)
    assert not small.any() and not complement.any() and ranges == {}


def test_linked_problem_protects_a_small_cell():
    p = MaskProblem(upper_bound=100)
    p.add("a", 2)
    p.add("b", 30)
    p.add("total", 32, fixed=True)
    p.equal("total", ["a", "b"])
    p.solve()
    assert p.shown("a") == MASK_SMALL and p.shown("b") == MASK_COMPLEMENT and p.check()["n_recoverable"] == 0
