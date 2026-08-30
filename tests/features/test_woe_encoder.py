import math

import numpy as np
import pandas as pd
import pytest

from credit_scoring.features.woeencoder import WOEEncoder


# --- toy fixtures ------------------------------------------------------------
# quantile toy: x = 0..7, n_bins=2 -> split at the median 3.5
#   bin (-inf, 3.5] : y = [0,0,0,1] -> good=3, bad=1
#   bin (3.5,  7.0] : y = [0,1,1,1] -> good=1, bad=3
#   all_good = all_bad = 4, smoothing = 0.5
#   woe(low)  = ln(((3+0.5)/4) / ((1+0.5)/4)) = ln(3.5/1.5) = ln(7/3)
#   woe(high) = ln(((1+0.5)/4) / ((3+0.5)/4)) = ln(1.5/3.5) = -ln(7/3)
X_TOY = pd.DataFrame({"x": [0, 1, 2, 3, 4, 5, 6, 7]})
Y_TOY = pd.Series([0, 0, 0, 1, 0, 1, 1, 1])
WOE_LOW = math.log(7 / 3)

# discrete toy: sentinel 98, tail threshold 2
#   rows:      d = [0, 0, 1, 1, 2, 5, 98, 98]
#              y = [0, 1, 0, 0, 1, 1,  0,  0]
#   all_good = 5, all_bad = 3, smoothing = 0.5
#   sentinel (d in {98}) : good=2, bad=0 -> ln((2.5/5)/(0.5/3)) = ln(3.0)
#   after sentinels are removed:
#   k=0                  : good=1, bad=1 -> ln((1.5/5)/(1.5/3)) = ln(0.6)
#   k=1                  : good=2, bad=0 -> ln((2.5/5)/(0.5/3)) = ln(3.0)
#   tail (d >= 2 -> 2,5) : good=0, bad=2 -> ln((0.5/5)/(2.5/3)) = ln(0.12)
X_DISC = pd.DataFrame({"d": [0, 0, 1, 1, 2, 5, 98, 98]})
Y_DISC = pd.Series([0, 1, 0, 0, 1, 1, 0, 0])


def fit_quantile_toy():
    return WOEEncoder(quantile_columns=["x"], discrete_columns=[], n_bins=2).fit(X_TOY, Y_TOY)


def fit_discrete_toy():
    return WOEEncoder(
        quantile_columns=[],
        discrete_columns=["d"],
        sentinel_values={"d": [98]},
        discrete_tail_threshold=2,
    ).fit(X_DISC, Y_DISC)


# --- quantile branch ---------------------------------------------------------
def test_quantile_bin_woe_matches_hand_computation():
    encoder = fit_quantile_toy()

    np.testing.assert_allclose(encoder.bin_edges_["x"], [0.0, 3.5, 7.0])

    woe_map = encoder.woe_maps_["x"]
    assert len(woe_map) == 2
    low, high = sorted(woe_map, key=lambda iv: iv.left)
    assert woe_map[low] == pytest.approx(WOE_LOW)
    assert woe_map[high] == pytest.approx(-WOE_LOW)


def test_quantile_transform_assigns_hand_computed_woe_per_row():
    result = fit_quantile_toy().transform(X_TOY)
    expected = [WOE_LOW] * 4 + [-WOE_LOW] * 4
    np.testing.assert_allclose(result["x"].astype(float).tolist(), expected)


# --- discrete branch ---------------------------------------------------------
def test_discrete_sentinel_gets_its_own_bin():
    woe_map = fit_discrete_toy().woe_maps_["d"]

    assert "sentinel" in woe_map
    assert woe_map["sentinel"] == pytest.approx(math.log(3.0))
    # the sentinel rows are pulled out before the regular bins are counted:
    # if 98 had leaked into "tail", tail would not be ln(0.12)
    assert woe_map["tail"] == pytest.approx(math.log(0.12))


def test_discrete_tail_covers_every_value_at_or_above_threshold():
    woe_map = fit_discrete_toy().woe_maps_["d"]

    # tail sees both d=2 and d=5 -> good=0, bad=2 -> ln(0.12)
    assert woe_map["tail"] == pytest.approx(math.log(0.12))
    # had the tail counted only the exact value 2, it would be ln(0.2)
    assert woe_map["tail"] != pytest.approx(math.log(0.2))


def test_discrete_regular_bins_match_hand_computation():
    woe_map = fit_discrete_toy().woe_maps_["d"]

    assert woe_map[0] == pytest.approx(math.log(0.6))
    assert woe_map[1] == pytest.approx(math.log(3.0))


def test_discrete_transform_assigns_hand_computed_woe_per_row():
    result = fit_discrete_toy().transform(X_DISC)
    expected = [
        math.log(0.6), math.log(0.6),      # d = 0
        math.log(3.0), math.log(3.0),      # d = 1
        math.log(0.12), math.log(0.12),    # d = 2, 5 -> tail
        math.log(3.0), math.log(3.0),      # d = 98  -> sentinel
    ]
    np.testing.assert_allclose(result["d"].astype(float).tolist(), expected)


# --- unseen values -----------------------------------------------------------
def test_unseen_quantile_value_falls_back_to_default_woe():
    encoder = fit_quantile_toy()  # fitted on x in [0, 7]
    result = encoder.transform(pd.DataFrame({"x": [100.0, -5.0]}))

    assert result["x"].notna().all()
    assert result["x"].tolist() == [encoder.default_woe_["x"]] * 2


def test_unseen_discrete_value_falls_back_to_default_woe():
    encoder = fit_discrete_toy()
    # -1 matches no sentinel, is below the tail threshold, and has no bin from fit
    result = encoder.transform(pd.DataFrame({"d": [-1]}))

    assert result["d"].notna().all()
    assert result["d"].tolist() == [encoder.default_woe_["d"]]


def test_discrete_value_larger_than_seen_at_fit_lands_in_tail_bin():
    encoder = fit_discrete_toy()  # fit saw at most d=5 outside the sentinels
    result = encoder.transform(pd.DataFrame({"d": [999]}))

    # by design any value >= the threshold is tail, so this is not a default case
    assert result["d"].tolist() == [pytest.approx(encoder.woe_maps_["d"]["tail"])]


# --- smoothing ---------------------------------------------------------------
def test_smoothing_keeps_woe_finite_for_a_pure_quantile_bin():
    # bin (-inf, 1.5] is all good, bin (1.5, 3] is all bad -> both would be +-inf unsmoothed
    X = pd.DataFrame({"x": [0, 1, 2, 3]})
    y = pd.Series([0, 0, 1, 1])
    woe_map = WOEEncoder(quantile_columns=["x"], discrete_columns=[], n_bins=2).fit(X, y).woe_maps_["x"]

    values = [float(v) for v in woe_map.values()]
    assert all(np.isfinite(values))
    # all_good = all_bad = 2, smoothing 0.5 -> ln((2.5/2)/(0.5/2)) = ln(5)
    assert max(values) == pytest.approx(math.log(5.0))
    assert min(values) == pytest.approx(-math.log(5.0))


def test_smoothing_keeps_woe_finite_for_pure_discrete_bins():
    # in the discrete toy: bin 1 has 0 bad, tail has 0 good
    woe_map = fit_discrete_toy().woe_maps_["d"]

    assert all(np.isfinite(float(v)) for v in woe_map.values())


# --- state is reused, not recomputed -----------------------------------------
def test_transform_uses_saved_edges_and_maps_not_recomputed_on_new_data():
    encoder = fit_quantile_toy()  # edges [0, 3.5, 7]
    edges_after_fit = encoder.bin_edges_["x"].copy()
    maps_after_fit = dict(encoder.woe_maps_["x"])

    # a frame whose own median is 1.5 - recomputed bins would split 0,1 | 2,3
    unseen = pd.DataFrame({"x": [0, 1, 2, 3]})
    result = encoder.transform(unseen)

    # with the saved edges all four rows sit in the low bin
    np.testing.assert_allclose(result["x"].astype(float).tolist(), [WOE_LOW] * 4)
    np.testing.assert_allclose(encoder.bin_edges_["x"], edges_after_fit)
    assert encoder.woe_maps_["x"] == maps_after_fit


def test_transform_does_not_mutate_input_frame():
    encoder = fit_quantile_toy()
    before = X_TOY.copy(deep=True)
    result = encoder.transform(X_TOY)

    assert result is not X_TOY
    pd.testing.assert_frame_equal(X_TOY, before)
