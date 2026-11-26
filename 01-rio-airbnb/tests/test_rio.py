"""Checks on the data, the split and the guardrails against leakage."""

import numpy as np
import pytest

import lesson
from rio import BOOKING, NUMERIC, load_listings, split_by_host


@pytest.fixture(scope="module")
def d():
    d = load_listings()
    d["split"] = split_by_host(d)
    return d


def test_every_host_stays_on_one_side_of_the_split(d):
    assert (d.groupby("host_group")["split"].nunique() == 1).all()
    shares = d["split"].value_counts(normalize=True)
    assert abs(shares["train"] - 0.70) < 0.03 and abs(shares["test"] - 0.15) < 0.03


def test_no_feature_gives_away_the_answers(d):
    features = set(NUMERIC) | set(BOOKING)
    assert not any("listings" in f or "host_group" in f for f in features)   # business = host's listing count
    assert not features & {"price", "log_price", "business"}


def test_targets(d):
    assert np.allclose(d["log_price"], np.log(d["price"]))
    assert set(d["business"].unique()) == {0, 1}
    assert 0.3 < d["business"].mean() < 0.6


def test_beach_distances_make_sense(d):
    by = d.groupby("neighbourhood")[["km_ocean_beach", "km_bay_beach"]].median()
    assert by.loc["Copacabana", "km_ocean_beach"] < 0.5     # on an ocean beach
    assert by.loc["Botafogo", "km_bay_beach"] < 1.0         # on a bay beach...
    assert by.loc["Botafogo", "km_ocean_beach"] > 1.0       # ...but not an ocean one
    assert (d[["km_ocean_beach", "km_bay_beach", "km_metro"]] >= 0).all().all()


def test_best_price_model_beats_the_lookup_baseline(d):
    tr, va = d[d.split == "train"], d[d.split == "validation"]
    model = lesson.price_scores(va.log_price.to_numpy(), lesson.best_price_model(tr, alpha=100)(va))
    assert model["r2"] > lesson.baselines(tr, va)["both"]["r2"] + 0.1
