"""Unit tests for ``hpgs.metrics.concordance_correlation_coefficient``.

agreement panel). CCC is the cross-cell summary metric locked in
``docs/scope_pr7.md`` § 3 (Q2 = C); unlike Pearson r it penalises
constant offsets and scale changes.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from hpgs.metrics import concordance_correlation_coefficient as ccc


def test_ccc_perfect_agreement_is_one() -> None:
    rng = np.random.default_rng(20260415)
    x = rng.normal(size=200)
    assert ccc(x, x) == pytest.approx(1.0, abs=1e-12)


def test_ccc_perfect_mirror_is_minus_one() -> None:
    # y = 2*mean(x) - x is the unique transformation that gives ccc = -1
    # while preserving mean(y) = mean(x) and var(y) = var(x).
    rng = np.random.default_rng(0xC0FFEE)
    x = rng.normal(loc=3.0, scale=2.0, size=500)
    y = 2.0 * x.mean() - x
    assert ccc(x, y) == pytest.approx(-1.0, abs=1e-12)


def test_ccc_constant_offset_penalised_relative_to_pearson() -> None:
    # Pearson r between x and x+c is exactly 1 for any c, but CCC must
    # be strictly less than 1 once c != 0 because the means disagree.
    rng = np.random.default_rng(42)
    x = rng.normal(size=300)
    y = x + 2.0  # constant offset
    value = ccc(x, y)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    assert pearson_r == pytest.approx(1.0, abs=1e-12)
    assert -1.0 < value < 1.0
    assert value < pearson_r


def test_ccc_scale_change_penalised_relative_to_pearson() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(size=300)
    y = 3.0 * x  # perfectly correlated, but different variance
    value = ccc(x, y)
    pearson_r = float(np.corrcoef(x, y)[0, 1])
    assert pearson_r == pytest.approx(1.0, abs=1e-12)
    assert value < pearson_r
    assert value > 0.0  # still strongly positive --- the *direction* agrees


def test_ccc_known_small_case() -> None:
    # Hand-computed reference value to lock the formula:
    #   x = [1, 2, 3, 4]   mean=2.5  var=1.25  (population, ddof=0)
    #   y = [1, 2, 3, 5]   mean=2.75 var~2.1875
    #   cov(x,y) = mean((x-mx)(y-my)) = mean([1.5*1.75, 0.5*0.75, 0.5*0.25, 1.5*2.25])
    #            = mean([2.625, 0.375, 0.125, 3.375]) = 6.5/4 = 1.625
    #   ccc = 2*1.625 / (1.25 + 2.1875 + (2.5 - 2.75)^2)
    #       = 3.25 / 3.5  ~ 0.928571
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 5.0])
    expected = 3.25 / (1.25 + 2.1875 + 0.0625)
    assert ccc(x, y) == pytest.approx(expected, abs=1e-12)


def test_ccc_drops_nan_pairs() -> None:
    # Inserting a NaN in either coordinate should not change the result
    # if the corresponding pair is dropped.
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = np.array([1.0, 2.0, 3.0, 5.0])
    expected = ccc(x, y)
    x_with_nan = np.array([1.0, 2.0, 3.0, 4.0, np.nan, 99.0])
    y_with_nan = np.array([1.0, 2.0, 3.0, 5.0, 99.0, np.nan])
    assert ccc(x_with_nan, y_with_nan) == pytest.approx(expected, abs=1e-12)


def test_ccc_keeps_nan_when_drop_nan_false() -> None:
    x = np.array([1.0, 2.0, np.nan])
    y = np.array([1.0, 2.0, 3.0])
    assert math.isnan(ccc(x, y, drop_nan=False))


def test_ccc_too_few_observations_returns_nan() -> None:
    assert math.isnan(ccc(np.array([1.0]), np.array([1.0])))
    assert math.isnan(ccc(np.array([np.nan, np.nan]), np.array([1.0, 2.0])))


def test_ccc_constant_inputs_return_nan() -> None:
    # Both variances zero, no signal to agree on -> nan.
    x = np.array([3.0, 3.0, 3.0])
    y = np.array([3.0, 3.0, 3.0])
    assert math.isnan(ccc(x, y))


def test_ccc_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        ccc(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0]))


def test_ccc_handles_2d_input_by_flattening() -> None:
    rng = np.random.default_rng(123)
    x = rng.normal(size=(20, 5))
    y = x.copy()
    assert ccc(x, y) == pytest.approx(1.0, abs=1e-12)
    assert ccc(x.ravel(), y.ravel()) == pytest.approx(ccc(x, y), abs=1e-12)
