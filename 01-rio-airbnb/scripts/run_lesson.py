"""Run every lesson, save the charts in figures/ and the numbers in reports/results.json.

    python scripts/run_lesson.py              # uses data/rio_listings.csv.gz (committed)
    python scripts/run_lesson.py --download   # re-download the raw sources and rebuild it first

The test set is scored once, at the end, with the choices made on the validation set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402

import lesson  # noqa: E402
from rio import DATA, build_listings, download, load_listings, save_slim, split_by_host  # noqa: E402

THRESHOLD = 0.7   # business host: few false alarms (chosen in step 6)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--download", action="store_true")
    args = p.parse_args()
    if args.download:
        download()
        save_slim(build_listings())
    d = load_listings()
    d["split"] = split_by_host(d)
    tr, va, te = (d[d.split == s] for s in ("train", "validation", "test"))
    print(f"{len(d):,} listings: {len(tr):,} train, {len(va):,} validation, {len(te):,} test")

    charts: dict = {}
    res = {"counts": {"listings": len(d), "train": len(tr), "validation": len(va), "test": len(te)}}
    lesson.explore(d, DATA / "neighbourhoods.geojson", save="01_explore.png")
    res["step1_beaches"] = lesson.beaches(tr, save="02_beaches.png", charts=charts)
    res["step2_baselines"] = lesson.baselines(tr, va)
    res["step3_one_feature"] = lesson.gradient_descent_one_feature(tr, va, save="03_gradient_descent.png", charts=charts)
    res["step4_all_features"] = lesson.scaling_and_features(tr, va, save="04_scaling.png", charts=charts)
    res["step5_overfitting"] = lesson.overfitting(tr, va, save="05_overfitting.png", charts=charts)
    res["step6_classification"] = lesson.classification(tr, va, THRESHOLD, save="06_classification.png", charts=charts)

    # Final test, once, with the choices made above.
    alpha = min(res["step5_overfitting"]["best_model_alphas"], key=lambda r: -r["val"]["r2"])["alpha"]
    price = lesson.best_price_model(tr, alpha)
    log_pred = price(te)
    business, _ = lesson.classifier(tr, booking=True)
    p_te = business(te)
    res["test"] = {
        "price_model": {"alpha": alpha, **lesson.price_scores(te.log_price.to_numpy(), log_pred)},
        "price_predictions": lesson.test_predictions(te, log_pred, save="07_test_predictions.png", charts=charts),
        "price_baseline_both_lookup": lesson.baselines(tr, te)["both"],
        "business": {"auc": round(float(lesson.roc_auc_score(te.business, p_te)), 3),
                     "always_not_business_accuracy": round(1 - float(te.business.mean()), 3),
                     "at_0.5": lesson.class_scores(te.business.to_numpy(), p_te, 0.5),
                     "at_chosen": lesson.class_scores(te.business.to_numpy(), p_te, THRESHOLD)},
    }
    out = ROOT / "reports"
    out.mkdir(exist_ok=True)
    (out / "results.json").write_text(json.dumps(res, indent=1, default=lambda o: float(o) if isinstance(o, np.floating) else str(o)) + "\n")
    (out / "charts.json").write_text(json.dumps(charts, separators=(",", ":")) + "\n")   # data for the website charts
    print(json.dumps(res["test"], indent=1))


if __name__ == "__main__":
    main()
