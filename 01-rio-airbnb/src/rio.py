"""Airbnb listings in Rio de Janeiro, with location features, ready for modelling.

Data:
- Listings: Inside Airbnb, Rio de Janeiro snapshot of 22 September 2026 (CC BY 4.0).
- Beaches and metro stations: OpenStreetMap (© OpenStreetMap contributors, ODbL), via the Overpass API.

Two targets:
- log_price: the natural log of the nightly price in reais (regression).
- business: 1 if the host runs 3 or more listings, i.e. uses Airbnb as a source of income (classification).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neighbors import BallTree

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SLIM = DATA / "rio_listings.csv.gz"   # committed: modelling columns only, hosts anonymised
LISTINGS_URL = "https://data.insideairbnb.com/brazil/rj/rio-de-janeiro/2026-09-22/data/listings.csv.gz"
OVERPASS = "https://overpass-api.de/api/interpreter"
OSM_QUERIES = {
    "osm_metro.json": '[out:json][timeout:90];area["name"="Rio de Janeiro"]["admin_level"="8"]->.a;'
                      '(node["station"="subway"](area.a);node["railway"="station"]["subway"="yes"](area.a););out;',
    "osm_beaches.json": '[out:json][timeout:120];area["name"="Rio de Janeiro"]["admin_level"="8"]->.a;'
                        '(way["natural"="beach"](area.a);relation["natural"="beach"](area.a););out geom;',
}
SNAPSHOT = pd.Timestamp("2026-09-22")
BUSINESS_MIN_LISTINGS = 3
SEED = 42

# Beaches inside Guanabara Bay or Sepetiba Bay: calm water, mostly not for swimming.
BAY_BEACHES = {
    "Praia de Botafogo", "Praia do Flamengo", "Praia da Urca", "Praia da Esplanada Verde", "Praia de Dentro",
    "Praia de Fora", "Praia de Ramos", "Praia Belo Jardim", "Praia da Bica", "Praia da Engenhoca",
    "Praia da Guanabara", "Praia do Bananal", "Praia da Batata", "Praia da Cidade", "Praia dos Frades",
    "Praia da Moreninha", "Praia de São Roque", "Praia da Guarda", "Praia dos Coqueiros", "Praia do Lameirão",
    "Praia dos Tamoios", "Praias de Moema e Iracema", "Praia Grossa", "Praia da Imbuca", "Praia do Buraco",
    "Praia do Catimbau", "Praia das Gaivotas", "Praia de Sepetiba", "Praia do Recôncavo", "Praia de Ponta Grossa",
}
NOT_BEACHES = {"Cossetti's Playground", "parquinho"}  # sand playgrounds tagged as beaches
SEA_VIEW = {"Beach view", "Ocean view", "Sea view"}
EARTH_KM = 6371.0


def _beach_points() -> dict[str, np.ndarray]:
    """Every vertex of every beach outline, as (lat, lon), split into ocean and bay beaches.

    Unnamed beaches are classed by the nearest named one."""
    elements = json.load(open(DATA / "osm_beaches.json"))["elements"]
    named, unnamed = [], []
    for e in elements:
        pts = [(g["lat"], g["lon"]) for g in e.get("geometry", [])] or \
              [(p["lat"], p["lon"]) for m in e.get("members", []) for p in m.get("geometry", [])]
        name = e.get("tags", {}).get("name")
        if not pts or name in NOT_BEACHES:
            continue
        (named if name else unnamed).append((name, np.array(pts)))
    centres = np.array([p.mean(axis=0) for _, p in named])
    kinds = {"ocean": [], "bay": []}
    for name, pts in named:
        kinds["bay" if name in BAY_BEACHES else "ocean"].append(pts)
    for _, pts in unnamed:
        nearest = named[int(np.argmin(((centres - pts.mean(axis=0)) ** 2).sum(axis=1)))][0]
        kinds["bay" if nearest in BAY_BEACHES else "ocean"].append(pts)
    return {k: np.vstack(v) for k, v in kinds.items()}


def _metro_points() -> np.ndarray:
    elements = json.load(open(DATA / "osm_metro.json"))["elements"]
    return np.array([(e["lat"], e["lon"]) for e in elements])


def nearest_km(lat: np.ndarray, lon: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Great-circle distance in km from each (lat, lon) to the nearest of `points`."""
    tree = BallTree(np.radians(points), metric="haversine")
    dist, _ = tree.query(np.radians(np.c_[lat, lon]), k=1)
    return dist[:, 0] * EARTH_KM


def download() -> None:
    """Fetch the raw sources: Inside Airbnb listings and OpenStreetMap beaches and metro stations."""
    import time

    import requests

    DATA.mkdir(exist_ok=True)
    (DATA / "listings.csv.gz").write_bytes(requests.get(LISTINGS_URL, timeout=300).content)
    headers = {"User-Agent": "ml-foundations-learning-project/0.1 (python data analysis)"}
    for name, query in OSM_QUERIES.items():
        r = requests.post(OVERPASS, data={"data": query}, headers=headers, timeout=300)
        r.raise_for_status()
        (DATA / name).write_bytes(r.content)
        time.sleep(5)  # be polite to the shared Overpass server


def build_listings() -> pd.DataFrame:
    """From the raw sources: listings with a usable price, the features and both targets."""
    raw = pd.read_csv(DATA / "listings.csv.gz")
    d = pd.DataFrame({"id": raw["id"]})
    d["host_group"] = pd.factorize(raw["host_id"])[0]  # anonymous: only used to keep a host's listings together
    d["price"] = pd.to_numeric(raw["price"].astype(str).str.replace(r"[$,]", "", regex=True), errors="coerce")
    d["neighbourhood"] = raw["neighbourhood_cleansed"]
    d["latitude"], d["longitude"] = raw["latitude"], raw["longitude"]

    # Location
    beaches = _beach_points()
    d["km_ocean_beach"] = nearest_km(d.latitude, d.longitude, beaches["ocean"])
    d["km_bay_beach"] = nearest_km(d.latitude, d.longitude, beaches["bay"])
    d["km_metro"] = nearest_km(d.latitude, d.longitude, _metro_points())
    amenities = raw["amenities"].fillna("[]").map(lambda s: set(json.loads(s)))
    d["sea_view"] = amenities.map(lambda a: bool(a & SEA_VIEW)).astype(int)
    d["beachfront"] = amenities.map(lambda a: any("Beachfront" in x for x in a)).astype(int)

    # The place
    d["room_type"] = raw["room_type"]
    d["guests"] = raw["accommodates"]
    d["bedrooms"] = raw["bedrooms"]
    baths = raw["bathrooms_text"].fillna("").str.lower()
    d["bathrooms"] = pd.to_numeric(baths.str.extract(r"([\d.]+)")[0], errors="coerce")
    d.loc[baths.str.contains("half"), "bathrooms"] = d["bathrooms"].fillna(0.5)
    d["shared_bath"] = baths.str.contains("shared").astype(int)

    # Reviews and age on Airbnb (first review as a proxy: no reviews yet = new listing)
    d["rating"] = raw["review_scores_rating"]
    d["no_reviews"] = d["rating"].isna().astype(int)
    first = pd.to_datetime(raw["first_review"])
    d["years_listed"] = ((SNAPSHOT - first).dt.days / 365.25).fillna(0.0)
    d["reviews"] = raw["number_of_reviews"]

    # Booking behaviour (for the business-host question)
    d["days_available"] = raw["availability_365"]
    d["min_nights"] = raw["minimum_nights"].clip(upper=365)
    d["instant_book"] = (raw["instant_bookable"] == "t").astype(int)
    d["superhost"] = (raw["host_is_superhost"] == "t").astype(int)

    # Targets. The host's listing count is the answer for `business`, so it is never a feature.
    d["log_price"] = np.log(d["price"])
    d["business"] = (raw["calculated_host_listings_count"] >= BUSINESS_MIN_LISTINGS).astype(int)

    d = d.dropna(subset=["price", "bedrooms", "bathrooms"])
    lo, hi = d["price"].quantile([0.01, 0.99])
    return d[d["price"].between(lo, hi)].reset_index(drop=True)


def save_slim(d: pd.DataFrame) -> None:
    d.to_csv(SLIM, index=False, float_format="%.6g", compression={"method": "gzip", "mtime": 0})


def load_listings() -> pd.DataFrame:
    """The modelling table: from the committed slim file, or rebuilt from the raw sources."""
    if SLIM.exists():
        return pd.read_csv(SLIM)
    d = build_listings()
    save_slim(d)
    return d


def split_by_host(d: pd.DataFrame, seed: int = SEED) -> pd.Series:
    """70% train, 15% validation, 15% test, keeping every host's listings on the same side.

    Otherwise a big host's other flats in training would give away its listings in the test set."""
    split = pd.Series("train", index=d.index)
    gss = GroupShuffleSplit(n_splits=1, test_size=0.30, random_state=seed)
    _, rest = next(gss.split(d, groups=d["host_group"]))
    rest_d = d.iloc[rest]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=0.50, random_state=seed)
    val, test = next(gss2.split(rest_d, groups=rest_d["host_group"]))
    split.iloc[rest[val]] = "validation"
    split.iloc[rest[test]] = "test"
    return split


NUMERIC = ["km_ocean_beach", "km_bay_beach", "km_metro", "sea_view", "beachfront", "guests", "bedrooms",
           "bathrooms", "shared_bath", "rating", "no_reviews", "years_listed", "reviews"]


def numeric_matrix(d: pd.DataFrame, rating_fill: float) -> pd.DataFrame:
    """Numeric features plus room-type indicators. Missing ratings (no reviews yet) get the training
    median; `no_reviews` marks them."""
    X = d[NUMERIC].copy()
    X["rating"] = X["rating"].fillna(rating_fill)
    # Whole home/apt is the reference. The handful of hotel rooms count as private rooms: a column
    # that is almost always 0 would be stretched to huge values by scaling.
    X["room_private"] = d["room_type"].isin(["Private room", "Hotel room"]).astype(int)
    X["room_shared"] = (d["room_type"] == "Shared room").astype(int)
    return X


BOOKING = ["days_available", "min_nights", "instant_book", "superhost"]
# Host response and acceptance rates are empty in this snapshot, so they are not used.


def booking_matrix(d: pd.DataFrame, min_nights_fill: float) -> pd.DataFrame:
    """Booking-behaviour features; the few missing minimum-night values get the training median."""
    X = d[BOOKING].copy()
    X["min_nights"] = X["min_nights"].fillna(min_nights_fill)
    return X
