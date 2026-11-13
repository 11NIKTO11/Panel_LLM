"""The statistical layer on small synthetic data: invariants of the transforms, and the tests against direct formulas."""
import numpy as np
import pandas as pd
import pytest

from panel.analysis.stats import classifier, mapping, permanova
from panel.analysis.stats.compositional import helmert_basis, ilr, replace_zeros
from panel.analysis.stats.design import block, indicators
from panel.analysis.stats.manova import DF, ETA, P, PILLAI, manova

RNG = np.random.default_rng(0)


def synthetic(n: int = 120, separated: bool = True):
    """Subjects with a four-level group, a number and a multi-answer column, and shares over four parts that do (or do not) follow the group."""
    group = pd.Series(RNG.choice(list("abcd"), n), name="group")
    shares = RNG.dirichlet(np.ones(4), n)
    if separated:
        shares[group == "a"] = RNG.dirichlet([6, 1, 1, 1], int((group == "a").sum()))
    subjects = pd.DataFrame({"group": group, "number": RNG.normal(size=n), "held": RNG.choice(["x", "y", "x; y", "z"], n)})
    return subjects, pd.DataFrame(shares, columns=list("wxyz"))


def test_helmert_basis_is_orthonormal_and_orthogonal_to_ones():
    basis = helmert_basis(5)
    assert basis.shape == (5, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(np.ones(5) @ basis, 0, atol=1e-12)
    with pytest.raises(ValueError):
        helmert_basis(1)


def test_zero_replacement_keeps_the_sum_and_the_other_ratios():
    replaced = replace_zeros(np.array([[0.5, 0.5, 0.0, 0.0], [0.2, 0.3, 0.5, 0.0]]), delta=0.005)
    np.testing.assert_allclose(replaced.sum(axis=1), 1)
    np.testing.assert_allclose(replaced[0], [0.495, 0.495, 0.005, 0.005])
    assert replaced[1, 0] / replaced[1, 1] == pytest.approx(0.2 / 0.3)
    with pytest.raises(ValueError):
        replace_zeros(np.array([[0.0, 0.0]]))
    assert ilr(np.array([[0.25, 0.25, 0.25, 0.25]])).shape == (1, 3)
    np.testing.assert_allclose(ilr(np.array([[0.25, 0.25, 0.25, 0.25]])), 0, atol=1e-12)


def test_design_blocks():
    subjects, _ = synthetic()
    assert block(subjects["number"]).shape == (120, 1)
    assert block(subjects["group"]).shape == (120, 3)
    assert block(subjects["held"], "; ").shape == (120, 2)
    with pytest.raises(ValueError):
        block(pd.Series(["a"] * 5, name="constant"))


def test_manova_finds_the_predictor_that_matters():
    subjects, shares = synthetic()
    table = manova(subjects, shares, ["group", "number"])
    assert list(table.index) == ["group", "number"] and table.loc["group", DF] == 3 and table.loc["number", DF] == 1
    assert table.loc["group", P] < 0.001 < table.loc["number", P]
    assert table.loc["number", ETA] == pytest.approx(table.loc["number", PILLAI])
    assert table.loc["group", ETA] == pytest.approx(table.loc["group", PILLAI] / 3)
    with pytest.raises(ValueError):
        manova(subjects.head(6), shares.head(6), ["group", "number"])


def test_permanova_matches_the_between_group_sums_of_squares():
    subjects, shares = synthetic()
    points = np.sqrt(shares.to_numpy()) / np.sqrt(2)
    total = ((points - points.mean(axis=0)) ** 2).sum()
    between = sum(len(rows) * ((points[rows].mean(axis=0) - points.mean(axis=0)) ** 2).sum() for rows in subjects.groupby("group").indices.values())
    centred = permanova.gower(permanova.pairwise(shares.to_numpy(), "hellinger"))
    degrees, share, ratio, chance = permanova.permanova(centred, block(subjects["group"]), permutations=199)
    assert degrees == 3 and share == pytest.approx(between / total) and chance == pytest.approx(1 / 200)
    assert ratio == pytest.approx((between / 3) / ((total - between) / (len(points) - 4)))
    assert permanova.permanova(centred, block(subjects["number"]), permutations=199)[3] > 0.01


@pytest.mark.parametrize("distance", permanova.DISTANCES)
def test_pairwise_distances_are_distances(distance):
    _, shares = synthetic(n=12)
    found = permanova.pairwise(shares.to_numpy(), distance)
    np.testing.assert_allclose(found, found.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(found), 0, atol=1e-7)
    assert (found >= 0).all()


def test_correlation_distance_is_between_two_rows():
    _, shares = synthetic(n=6)
    values = shares.to_numpy()
    found = permanova.pairwise(values, "correlation")
    assert found[0, 1] == pytest.approx(np.sqrt(2 * (1 - np.corrcoef(values[0], values[1])[0, 1])))
    with pytest.raises(ValueError):
        permanova.pairwise(values, "nope")


def test_a_turned_map_fits_its_original():
    target = RNG.normal(size=(6, 2))
    turn = np.array([[0.0, -1.0], [1.0, 0.0]])
    fitted, r = mapping.procrustes(target, 3 * target @ turn + 5)
    np.testing.assert_allclose(fitted, target, atol=1e-9)
    assert r == pytest.approx(1)
    observed, chance = mapping.procrustes_test(target, 3 * target @ turn + 5)
    assert observed == pytest.approx(1) and chance == pytest.approx(1 / 720)


def test_mds_and_mantel_on_points_of_a_plane():
    points = RNG.normal(size=(6, 2))
    distances = np.linalg.norm(points[:, None] - points[None], axis=-1)
    placed, stress = mapping.mds(distances)
    assert placed.shape == (6, 2) and stress < 0.01
    observed, chance = mapping.mantel(distances, np.linalg.norm(placed[:, None] - placed[None], axis=-1))
    assert observed > 0.99 and chance == pytest.approx(1 / 720)
    with pytest.raises(ValueError):
        mapping.mantel(np.zeros((9, 9)), np.zeros((9, 9)))


def test_indicators_keep_every_level():
    subjects, _ = synthetic()
    assert indicators(subjects["group"]).shape == (120, 4)
    assert indicators(subjects["held"], "; ").shape == (120, 3)
    assert indicators(pd.Series(["a", None, "b"], name="gaps")).shape == (3, 3)
    banded = indicators(pd.Series([20, 30, 70], name="age"), bands={"young": 25, "old": 99})
    np.testing.assert_allclose(banded, [[1, 0], [0, 1], [0, 1]])
    with pytest.raises(ValueError):
        indicators(pd.Series([20, 120], name="age"), bands={"young": 25, "old": 99})


def test_a_trained_model_finds_what_decides_the_answer():
    features = RNG.normal(size=(200, 2))
    answers = np.where(features[:, 0] > 0, "x", "y")
    found = classifier.cross_validated(features, answers, ["x", "y", "z"], strength=1.0)
    answered = pd.get_dummies(answers).reindex(columns=["x", "y", "z"], fill_value=0).to_numpy(dtype=float)
    np.testing.assert_allclose(found.sum(axis=1), 1)
    assert (found[:, 2] == 0).all() and (found.argmax(axis=1) == answered.argmax(axis=1)).mean() > 0.9
    known, blind = classifier.losses(found, answered)[0].mean(), classifier.losses(np.full((200, 3), 1 / 3), answered)[0].mean()
    assert known < blind == pytest.approx(np.log(3))
    assert classifier.losses(answered, answered, floor=0.0)[1].max() == 0
    assert classifier.recalibrated(found, answers, ["x", "y", "z"]).shape == (200, 3)
