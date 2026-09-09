"""The statistical layer on small synthetic data: invariants of the transforms and the direction of the tests."""
import numpy as np
import pandas as pd
import pytest

from panel.analysis.stats.compositional import helmert_basis, ilr, ilr_frame
from panel.analysis.stats.manova import effects, manova
from panel.analysis.stats.permanova import DISTANCES, pairwise_distances, panel_permanova, permanova

RNG = np.random.default_rng(0)


def synthetic(n: int = 60, separated: bool = True):
    """A profile with one binary predictor and distributions over three options that do (or do not) depend on it."""
    group = np.array(["a", "b"] * (n // 2))
    base = RNG.dirichlet([2, 2, 2], size=n)
    shift = np.where((group == "b")[:, None], np.array([0.4, -0.2, -0.2]), 0.0) if separated else 0.0
    values = np.clip(base + shift, 0.01, None)
    values = values / values.sum(axis=1, keepdims=True)
    index = pd.Index(range(1, n + 1), name="id")
    profile = pd.DataFrame({"group": group, "noise": RNG.normal(size=n)}, index=index)
    results = pd.DataFrame(values, columns=["x", "y", "z"], index=index)
    return profile, results


def test_helmert_basis_is_orthonormal_and_orthogonal_to_ones():
    basis = helmert_basis(5)
    assert basis.shape == (5, 4)
    np.testing.assert_allclose(basis.T @ basis, np.eye(4), atol=1e-12)
    np.testing.assert_allclose(np.ones(5) @ basis, 0, atol=1e-12)
    with pytest.raises(ValueError):
        helmert_basis(1)


def test_ilr_shapes_and_frame():
    compositions = np.array([[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]])
    coordinates = ilr(compositions)
    assert coordinates.shape == (2, 2)
    frame = pd.DataFrame(compositions, columns=["x", "y", "z"])
    out = ilr_frame(frame, ["x", "y", "z"])
    assert list(out.columns) == ["x", "y", "z", "ilr_1", "ilr_2"]
    np.testing.assert_allclose(out[["ilr_1", "ilr_2"]].to_numpy(), coordinates)
    with pytest.raises(KeyError):
        ilr_frame(frame, ["x", "missing"])


@pytest.mark.parametrize("metric", DISTANCES)
def test_pairwise_distances_are_metrics(metric):
    points = RNG.dirichlet([1, 1, 1], size=6)
    space = ilr(points) if metric == "euclidean_ilr" else points
    distance = pairwise_distances(space, metric)
    assert distance.shape == (6, 6)
    np.testing.assert_allclose(distance, distance.T, atol=1e-12)
    np.testing.assert_allclose(np.diag(distance), 0, atol=1e-6)
    assert (distance >= -1e-12).all()
    with pytest.raises(ValueError):
        pairwise_distances(points, "nonsense")


def test_permanova_detects_separation():
    profile, results = synthetic(separated=True)
    result, report = panel_permanova(profile, results, ["group"], ["x", "y", "z"], n_perm=199, seed=1)
    assert result.p_value < 0.05 and result.df == 1 and report["n_obs"] == 60 and report["n_predictors"] == 1
    profile, results = synthetic(separated=False)
    result, _ = panel_permanova(profile, results, ["group"], ["x", "y", "z"], distance="hellinger", n_perm=199, seed=1)
    assert result.p_value > 0.05
    with pytest.raises(ValueError):
        panel_permanova(profile, results, ["missing"], ["x", "y", "z"])


def test_permanova_p_value_bounds():
    distance = pairwise_distances(RNG.normal(size=(12, 3)), "euclidean_ilr")
    design = RNG.normal(size=(12, 2))
    result = permanova(distance, design, n_perm=50, seed=3)
    assert 0 < result.p_value <= 1 and result.n_perm == 50 and result.ss >= 0


def test_manova_and_effects():
    profile, results = synthetic(separated=True)
    tests = manova(profile, results, ["group"], ["x", "y", "z"])
    summary = effects({"synthetic": tests})
    assert list(summary.index) == [("synthetic", "group")]
    assert 0 < summary.loc[("synthetic", "group"), "Pillai's trace"] <= 1
    assert 0 < summary.loc[("synthetic", "group"), "Wilks' lambda"] < 1
    with pytest.raises(ValueError):
        manova(profile.iloc[:0], results, ["group"], ["x", "y", "z"])
