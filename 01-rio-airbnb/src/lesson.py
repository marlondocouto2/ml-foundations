"""The six lessons, as functions: each returns its numbers and (optionally) saves its chart.

Course 1 of the Machine Learning Specialization (regression and classification), practised with
scikit-learn on Airbnb listings in Rio de Janeiro. See `rio.py` for the data.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LogNorm  # noqa: E402
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, SGDRegressor  # noqa: E402
from sklearn.metrics import mean_squared_error, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler  # noqa: E402

from rio import ROOT, booking_matrix, numeric_matrix  # noqa: E402

warnings.filterwarnings("ignore", category=RuntimeWarning)
FIG = ROOT / "figures"
COLORS = {"blue": "#2a78d6", "orange": "#eb6834", "green": "#1baf7a", "violet": "#6250d6", "yellow": "#eda100",
          "gray": "#8a8984", "ink": "#0b0b0b", "ink2": "#52514e", "line": "#cfcdc6"}
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})


# ---------------------------------------------------------------- scoring helpers
def r2(y, p) -> float:
    return float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def typical_miss(y, p) -> float:
    """exp(mean |log error|) - 1: how far off a typical prediction is, as a percentage."""
    return float(np.expm1(np.abs(y - p).mean()) * 100)


def price_scores(y, p) -> dict:
    return {"r2": round(r2(y, p), 3), "typical_miss_pct": round(typical_miss(y, p), 1),
            "median_miss_brl": round(float(np.median(np.abs(np.exp(y) - np.exp(p)))))}


def _save(fig, name: str | None):
    if name:
        FIG.mkdir(exist_ok=True)
        fig.tight_layout()
        fig.savefig(FIG / name, dpi=140)
    return fig


# ---------------------------------------------------------------- step 1: explore
def explore(d: pd.DataFrame, neighbourhoods_geojson: Path | None = None, save: str | None = None):
    """Price map, host map, price distribution, listings per host."""
    import json
    fig = plt.figure(figsize=(15, 9.5))
    ax1, ax2, ax3, ax4 = (fig.add_subplot(2, 2, i) for i in range(1, 5))
    gj = json.load(open(neighbourhoods_geojson)) if neighbourhoods_geojson and neighbourhoods_geojson.exists() else None

    def outlines(ax):
        if not gj:
            return
        for f in gj["features"]:
            g = f["geometry"]
            for poly in (g["coordinates"] if g["type"] == "MultiPolygon" else [g["coordinates"]]):
                ring = np.array(poly[0])
                ax.plot(ring[:, 0], ring[:, 1], color=COLORS["line"], lw=0.4, zorder=1)

    for ax in (ax1, ax2):
        outlines(ax)
        ax.set_aspect(1 / np.cos(np.radians(-22.9)))
        ax.axis("off")
    sc = ax1.scatter(d.longitude, d.latitude, c=d.price, s=1.5, cmap="viridis", norm=LogNorm(100, 3000), zorder=2)
    fig.colorbar(sc, ax=ax1, label="Nightly price (R$, log scale)")
    ax1.set_title("Price per night")
    biz = d.business == 1
    ax2.scatter(d.longitude[~biz], d.latitude[~biz], s=1.5, color=COLORS["yellow"], label="regular host (1–2 listings)", zorder=2)
    ax2.scatter(d.longitude[biz], d.latitude[biz], s=1.5, color=COLORS["violet"], alpha=0.5, label="business host (3+ listings)", zorder=3)
    ax2.legend(markerscale=6, frameon=False, loc="lower left")
    ax2.set_title("Who hosts")
    bins = np.logspace(np.log10(80), np.log10(8000), 50)
    for rt, c in [("Entire home/apt", COLORS["blue"]), ("Private room", COLORS["orange"])]:
        ax3.hist(d.price[d.room_type == rt], bins=bins, alpha=0.6, color=c, label=rt)
    ax3.set_xscale("log")
    ax3.set(xlabel="Nightly price (R$, log scale)", ylabel="Listings", title="Prices are skewed: most are modest, a few are very expensive")
    ax3.legend(frameon=False)
    ax4.bar(["Regular host", "Business host"], [(~biz).mean() * 100, biz.mean() * 100], color=[COLORS["yellow"], COLORS["violet"]])
    ax4.set(ylabel="% of listings", title="Share of listings by host type")
    fig.suptitle(f"Airbnb in Rio de Janeiro, 22 September 2026: {len(d):,} listings. Data: Inside Airbnb (CC BY 4.0), "
                 "© OpenStreetMap contributors", x=0.01, ha="left", fontsize=9, color=COLORS["ink2"])
    return _save(fig, save)


def beaches(tr: pd.DataFrame, save: str | None = None) -> dict:
    """Median price by distance to the nearest ocean beach and bay beach; a few neighbourhoods marked."""
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), sharey=True)
    bins = [0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 25]
    out = {}
    for a, col, label, color in [(ax[0], "km_ocean_beach", "ocean beach", COLORS["blue"]),
                                 (ax[1], "km_bay_beach", "bay beach", COLORS["green"])]:
        g = tr.groupby(pd.cut(tr[col], bins), observed=True).price.median()
        a.plot([(i.left + i.right) / 2 for i in g.index], g.values, marker="o", color=color, lw=2)
        a.set_xscale("log")
        a.set_xticks([0.2, 0.5, 1, 2, 5, 10, 20])
        a.set_xticklabels(["0.2", "0.5", "1", "2", "5", "10", "20"])
        a.set(xlabel=f"Distance to the nearest {label} (km, log scale)", title=f"Price vs distance to the nearest {label}")
        for n, c in [("Botafogo", COLORS["orange"]), ("Copacabana", COLORS["violet"]), ("Tijuca", COLORS["gray"])]:
            s = tr[tr.neighbourhood == n]
            a.scatter(s[col].median(), s.price.median(), s=120, color=c, zorder=5, edgecolor="white")
            a.annotate(n, (s[col].median(), s.price.median()), xytext=(6, 6), textcoords="offset points", color=c, fontsize=9)
        out[col] = {str(k): round(float(v)) for k, v in g.items()}
    ax[0].set_ylabel("Median nightly price (R$)")
    _save(fig, save)
    return out


# ---------------------------------------------------------------- step 2: baselines
def baselines(tr: pd.DataFrame, va: pd.DataFrame) -> dict:
    size = lambda df: df.room_type.str.split().str[0] + "_" + df.bedrooms.clip(upper=5).astype(int).astype(str)  # noqa: E731
    tr, va = tr.assign(size=size(tr)), va.assign(size=size(va))
    overall = tr.log_price.median()

    def lookup(keys):
        g = tr.groupby(keys).log_price.agg(["median", "size"])
        g = g[g["size"] >= 20]["median"]
        idx = pd.MultiIndex.from_frame(va[keys]) if len(keys) > 1 else va[keys[0]]
        return pd.Series(np.asarray(idx.map(g.to_dict().get) if len(keys) > 1 else idx.map(g), dtype=float),
                         index=va.index).fillna(overall)

    y = va.log_price
    return {
        "typical Rio price": price_scores(y, pd.Series(overall, index=va.index)),
        "location only (neighbourhood)": price_scores(y, lookup(["neighbourhood"])),
        "size only (room type x bedrooms)": price_scores(y, lookup(["size"])),
        "both": price_scores(y, lookup(["neighbourhood", "size"])),
        "business: always 'not business' accuracy": round(1 - float(va.business.mean()), 3),
    }


# ---------------------------------------------------------------- step 3: gradient descent, one feature
def gradient_descent_one_feature(tr: pd.DataFrame, va: pd.DataFrame, save: str | None = None) -> dict:
    X, y = tr[["bedrooms"]].to_numpy(float), tr.log_price.to_numpy()
    best_cost = mean_squared_error(y, LinearRegression().fit(X, y).predict(X))
    runs = {}
    for eta, label, color in [(0.000005, "too small (0.000005)", COLORS["gray"]),
                              (0.0001, "about right (0.0001)", COLORS["green"]),
                              (0.3, "too large (0.3)", COLORS["orange"])]:
        m = SGDRegressor(penalty=None, learning_rate="constant", eta0=eta, random_state=0)
        costs = []
        for _ in range(30):
            m.partial_fit(X, y)
            costs.append(mean_squared_error(y, m.predict(X)))
        runs[label] = (costs, color, m)
    good = runs["about right (0.0001)"][2]
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    rng = np.random.default_rng(0)
    s = tr.sample(6000, random_state=0)
    ax[0].scatter(s.bedrooms.clip(upper=6) + rng.uniform(-0.25, 0.25, len(s)), s.price, s=3, alpha=0.2, color=COLORS["blue"])
    xs = np.arange(0, 7)
    ax[0].plot(xs, np.exp(good.intercept_[0] + good.coef_[0] * xs), color=COLORS["green"], lw=2.5, label="model learned by gradient descent")
    ax[0].set_yscale("log")
    ax[0].set_yticks([100, 200, 500, 1000, 2000, 5000])
    ax[0].set_yticklabels(["100", "200", "500", "1,000", "2,000", "5,000"])
    ax[0].set(xlabel="Bedrooms (6 = 6 or more)", ylabel="Nightly price (R$, log scale)",
              title=f"Each extra bedroom: about +{np.expm1(good.coef_[0]) * 100:.0f}% on the price")
    ax[0].legend(frameon=False, loc="upper left")
    for label, (costs, color, _) in runs.items():
        ax[1].plot(range(1, 31), costs, marker="o", ms=3, color=color, label=label)
    ax[1].axhline(best_cost, color=COLORS["ink"], ls="--", lw=1)
    ax[1].text(29.5, best_cost * 0.93, "lowest possible cost", ha="right", va="top", fontsize=9)
    ax[1].set_yscale("log")
    ax[1].set(xlabel="Passes through the training data", ylabel="Cost: average squared miss (log scale)",
              title="Gradient descent with three learning rates")
    ax[1].legend(frameon=False)
    _save(fig, save)
    p = good.predict(va[["bedrooms"]].to_numpy(float))
    return {"base_price_brl": round(float(np.exp(good.intercept_[0]))), "per_bedroom_pct": round(float(np.expm1(good.coef_[0]) * 100), 1),
            "validation": price_scores(va.log_price.to_numpy(), p)}


# ---------------------------------------------------------------- step 4: scaling and all features
def scaling_and_features(tr: pd.DataFrame, va: pd.DataFrame, save: str | None = None) -> dict:
    fill = tr.rating.median()
    Xtr, Xva = numeric_matrix(tr, fill), numeric_matrix(va, fill)
    y = tr.log_price.to_numpy()
    best = mean_squared_error(y, LinearRegression().fit(Xtr, y).predict(Xtr))
    Xs = StandardScaler().fit_transform(Xtr)

    def run(Xm, eta):
        m = SGDRegressor(penalty=None, learning_rate="constant", eta0=eta, random_state=0)
        out = []
        for _ in range(30):
            m.partial_fit(Xm, y)
            ok = np.all(np.isfinite(m.coef_)) and np.abs(m.coef_).max() < 1e8
            out.append(mean_squared_error(y, m.predict(Xm)) if ok else np.nan)
        return out

    runs = [("Scaled, learning rate 0.0001", run(Xs, 0.0001), COLORS["green"]),
            ("Raw, same learning rate 0.0001", run(Xtr.to_numpy(float), 0.0001), COLORS["orange"]),
            ("Raw, 10x smaller rate 0.00001", run(Xtr.to_numpy(float), 0.00001), COLORS["gray"])]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    for n, c, col in runs:
        ax.plot(range(1, 31), c, marker="o", ms=3, color=col, label=n)
    ax.axhline(best, color=COLORS["ink"], ls="--", lw=1)
    ax.text(29.5, best * 0.97, "lowest possible cost", ha="right", va="top", fontsize=9)
    ax.set(xlabel="Passes through the training data", ylabel="Cost: average squared miss",
           title="Same features, same gradient descent: with and without scaling", ylim=(best * 0.85, 2.0))
    ax.legend(frameon=False)
    _save(fig, save)

    m = LinearRegression().fit(Xtr, y)
    effects = pd.Series(np.expm1(m.coef_) * 100, index=Xtr.columns).round(1)
    ms = LinearRegression().fit(StandardScaler().fit_transform(Xtr), y)
    importance = pd.Series(np.abs(ms.coef_), index=Xtr.columns).round(3).sort_values(ascending=False)
    return {"scaling_final_cost": {n: round(c[-1], 3) for n, c, _ in runs}, "lowest_cost": round(best, 3),
            "validation": price_scores(va.log_price.to_numpy(), m.predict(Xva)),
            "effects_pct": effects.to_dict(), "importance_order": list(importance.index)}


# ---------------------------------------------------------------- step 5: curves, neighbourhoods, regularization
def _neigh_encoder(tr):
    return OneHotEncoder(handle_unknown="ignore", sparse_output=False, min_frequency=10).fit(tr[["neighbourhood"]])


def best_price_model(tr: pd.DataFrame, alpha: float):
    """Degree-2 curves of the numeric features plus neighbourhoods, all scaled, with Ridge.
    Scaling after the polynomial step gives every term an even penalty."""
    fill = tr.rating.median()
    poly = make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False)).fit(numeric_matrix(tr, fill))
    enc = _neigh_encoder(tr)

    def X(df):
        return np.hstack([poly.transform(numeric_matrix(df, fill)), enc.transform(df[["neighbourhood"]])])

    scale = StandardScaler().fit(X(tr))
    model = Ridge(alpha=alpha).fit(scale.transform(X(tr)), tr.log_price)
    return lambda df: model.predict(scale.transform(X(df)))


def overfitting(tr: pd.DataFrame, va: pd.DataFrame, save: str | None = None) -> dict:
    fill = tr.rating.median()
    Xtr, Xva = numeric_matrix(tr, fill), numeric_matrix(va, fill)
    ytr, yva = tr.log_price.to_numpy(), va.log_price.to_numpy()
    enc = _neigh_encoder(tr)
    m = LinearRegression().fit(np.c_[Xtr, enc.transform(tr[["neighbourhood"]])], ytr)
    with_neigh = {"train_r2": round(r2(ytr, m.predict(np.c_[Xtr, enc.transform(tr[["neighbourhood"]])])), 3),
                  "val_r2": round(r2(yva, m.predict(np.c_[Xva, enc.transform(va[["neighbourhood"]])])), 3)}
    degrees = []
    for deg in [1, 2, 3, 4]:
        m = make_pipeline(StandardScaler(), PolynomialFeatures(deg, include_bias=False), LinearRegression()).fit(Xtr, ytr)
        degrees.append({"degree": deg, "terms": int(m[1].n_output_features_), "train_r2": round(r2(ytr, m.predict(Xtr)), 3),
                        "val_r2": round(r2(yva, m.predict(Xva)), 3)})
    ridge = []
    for a in [0.01, 0.1, 1, 10, 100, 1000, 10000, 100000]:
        m = make_pipeline(StandardScaler(), PolynomialFeatures(3, include_bias=False), StandardScaler(), Ridge(alpha=a)).fit(Xtr, ytr)
        ridge.append({"alpha": a, "train_r2": round(r2(ytr, m.predict(Xtr)), 3), "val_r2": round(r2(yva, m.predict(Xva)), 3)})
    best = []
    for a in [1, 10, 100, 1000, 10000]:
        best.append({"alpha": a, "val": price_scores(yva, best_price_model(tr, a)(va))})
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6))
    dd = pd.DataFrame(degrees)
    ax[0].plot(dd.degree, dd.train_r2, marker="o", color=COLORS["blue"], label="training listings")
    ax[0].plot(dd.degree, dd.val_r2.clip(lower=-0.1), marker="o", color=COLORS["orange"], label="new listings (tuning set)")
    if dd.val_r2.iloc[-1] < -0.1:
        ax[0].annotate(f"{dd.val_r2.iloc[-1]:.0f}\n(off the chart)", (4, -0.1), xytext=(-70, 12), textcoords="offset points",
                       fontsize=9, color=COLORS["orange"])
    ax[0].set(xticks=[1, 2, 3, 4], xlabel="Polynomial degree (how much the model can bend)",
              ylabel="Share of price variation explained (R²)", title="Too much flexibility: great on training data, bad on new data",
              ylim=(-0.12, 0.7))
    ax[0].legend(frameon=False, loc="upper left")
    rr = pd.DataFrame(ridge)
    ax[1].plot(rr.alpha, rr.train_r2, marker="o", color=COLORS["blue"], label="training listings")
    ax[1].plot(rr.alpha, rr.val_r2.clip(lower=-0.1), marker="o", color=COLORS["orange"], label="new listings (tuning set)")
    ax[1].set_xscale("log")
    ax[1].set(xlabel="Regularization strength (Ridge alpha, log scale)", ylabel="R²",
              title="Degree 3, regularized: rescued, then over-restrained", ylim=(-0.12, 0.7))
    ax[1].legend(frameon=False, loc="lower left")
    _save(fig, save)
    return {"with_neighbourhoods": with_neigh, "degrees": degrees, "ridge_degree3": ridge, "best_model_alphas": best}


# ---------------------------------------------------------------- step 6: business host or not
def classifier(tr: pd.DataFrame, booking: bool = True):
    fill, mn = tr.rating.median(), tr.min_nights.median()
    enc = _neigh_encoder(tr)

    def X(df):
        parts = [numeric_matrix(df, fill).to_numpy(float), enc.transform(df[["neighbourhood"]])]
        if booking:
            parts.insert(1, booking_matrix(df, mn).to_numpy(float))
        return np.hstack(parts)

    scale = StandardScaler().fit(X(tr))
    model = LogisticRegression(max_iter=3000).fit(scale.transform(X(tr)), tr.business)
    names = list(numeric_matrix(tr, fill).columns) + (list(booking_matrix(tr, mn).columns) if booking else [])
    coefs = pd.Series(model.coef_[0][:len(names)], index=names)
    return (lambda df: model.predict_proba(scale.transform(X(df)))[:, 1]), coefs


def class_scores(y, p, threshold: float) -> dict:
    flag = p >= threshold
    return {"threshold": threshold, "accuracy": round(float((flag == y).mean()), 3), "flagged_pct": round(float(flag.mean() * 100), 1),
            "precision": round(float(precision_score(y, flag, zero_division=0)), 3), "recall": round(float(recall_score(y, flag)), 3)}


def classification(tr: pd.DataFrame, va: pd.DataFrame, threshold: float = 0.7, save: str | None = None) -> dict:
    y = va.business.to_numpy()
    out = {}
    for name, booking in [("price features", False), ("+ booking behaviour", True)]:
        predict, coefs = classifier(tr, booking)
        p = predict(va)
        out[name] = {"auc": round(float(roc_auc_score(y, p)), 3), "at_0.5": class_scores(y, p, 0.5)}
    out["thresholds"] = [class_scores(y, p, t) for t in [0.5, 0.6, 0.7, 0.8]]
    out["strongest_signs"] = coefs.reindex(coefs.abs().sort_values(ascending=False).index).round(2).head(6).to_dict()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    ts = np.linspace(0.3, 0.85, 56)
    ax[0].plot(ts, [precision_score(y, p >= t, zero_division=np.nan) for t in ts], color=COLORS["violet"], lw=2,
               label="Precision: of listings flagged, share truly business")
    ax[0].plot(ts, [recall_score(y, p >= t) for t in ts], color=COLORS["yellow"], lw=2, label="Recall: of business listings, share caught")
    ax[0].axvline(threshold, color=COLORS["ink"], ls="--", lw=1)
    ax[0].text(threshold + 0.005, 0.97, "chosen cut-off:\nfew false alarms", fontsize=9, va="top")
    ax[0].set(xlabel="Cut-off: call it 'business' when the probability is at least…", ylabel="Share", ylim=(0, 1),
              title="Fewer false alarms means catching fewer business hosts")
    ax[0].legend(frameon=False, loc="lower left", fontsize=9)
    fill = tr.rating.median()
    T, V = tr.assign(rating=tr.rating.fillna(fill)), va.assign(rating=va.rating.fillna(fill))
    s2 = StandardScaler().fit(T[["rating", "guests"]])
    m2 = LogisticRegression().fit(s2.transform(T[["rating", "guests"]]), T.business)
    smp = V.sample(2500, random_state=1)
    rng = np.random.default_rng(0)
    xj = smp.rating.clip(3.5, 5) + rng.uniform(-0.02, 0.02, len(smp))
    yj = smp.guests.clip(upper=10) + rng.uniform(-0.3, 0.3, len(smp))
    for lab, col, txt in [(0, COLORS["yellow"], "regular host (1–2 listings)"), (1, COLORS["violet"], "business host (3+ listings)")]:
        ax[1].scatter(xj[smp.business == lab], yj[smp.business == lab], s=6, alpha=0.4, color=col, label=txt)
    gx, gy = np.meshgrid(np.linspace(3.5, 5, 200), np.linspace(0.5, 10.5, 200))
    pp = m2.predict_proba(s2.transform(pd.DataFrame({"rating": gx.ravel(), "guests": gy.ravel()})))[:, 1].reshape(gx.shape)
    cs = ax[1].contour(gx, gy, pp, levels=[0.4, 0.5, 0.6], colors=[COLORS["gray"], COLORS["ink"], COLORS["gray"]],
                       linestyles=[":", "-", ":"])
    ax[1].clabel(cs, fmt={0.4: "40%", 0.5: "50%", 0.6: "60%"}, fontsize=8)
    ax[1].set(xlabel="Rating (stars; listings without reviews shown at the median)", ylabel="Guests it sleeps",
              title="Decision boundary on two features: the groups overlap heavily")
    ax[1].legend(frameon=False, loc="upper left", markerscale=3, fontsize=9)
    _save(fig, save)
    return out
