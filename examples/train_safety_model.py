"""
train_safety_model.py
----------------------
Trains a RandomForest classifier that learns to distinguish "safe" vs "unsafe"
driving patterns from CARLA telemetry logs (e.g. driving_data.csv).

WHY A HEURISTIC LABEL?
The raw CSV has no human-labeled "safe/unsafe" column. So this script first
derives a label from physically-grounded rules (harsh braking at speed,
collisions, lane invasions, excess steering at speed, low-visibility speeding,
wet-road hard braking, etc.), then trains a model on top of that label.

The point of the ML model is NOT to reinvent the rules -- it's to let the
dashboard get a smooth, continuous "risk probability" (instead of a hard
if/else), and to generalize to combinations of conditions that weren't
explicitly covered by any single rule (e.g. moderate speed + moderate rain +
moderate steering all at once).

USAGE:
    python train_safety_model.py driving_data.csv

Produces:
    safety_model.pkl   -> {"model": RandomForestClassifier, "features": [...]}

Re-run this any time you collect a new/bigger CSV (e.g. your future dataset)
-- the dashboard will pick up the new model automatically on next launch.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

FEATURES = [
    "speed_kmh", "throttle", "brake", "steer", "abs_steer",
    "rain", "fog", "accel", "brake_jerk", "brake_energy",
    "num_vehicles", "num_pedestrians",
    # Obstacle-radar / predictive-braking features (only present in CSVs
    # logged by the updated dashboard -- default to "no obstacle nearby" /
    # "no priming" for older-schema CSVs, see load_and_clean()).
    "obstacle_front_m", "obstacle_rear_m",
    "min_side_clearance_m", "brake_priming", "brake_actuated",
]

MODEL_PATH = "safety_model.pkl"


def load_and_clean(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    # Be tolerant of slightly different column sets/names across CSV versions.
    # Older-schema CSVs (logged before the obstacle-radar / predictive-braking
    # update) simply won't have those columns -- default them to "no obstacle
    # nearby" / "no priming" so the same script keeps working on both.
    defaults = {
        "speed_kmh": 0.0, "throttle": 0.0, "brake": 0.0, "steer": 0.0,
        "rain": 0, "fog": 0, "num_vehicles": 0, "num_pedestrians": 0,
        "collision": 0, "lane_invasion": 0,
        "obstacle_front_m": 30.0, "obstacle_rear_m": 30.0,
        "obstacle_left_m": 30.0, "obstacle_right_m": 30.0,
        "safe_braking_distance_m": 0.0, "brake_priming": 0.0,
        "brake_actuated": 0.0, "aeb_triggered": 0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    df["min_side_clearance_m"] = df[["obstacle_left_m", "obstacle_right_m"]].min(axis=1)

    df = df.sort_values("timestamp") if "timestamp" in df.columns else df
    return df.reset_index(drop=True)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["abs_steer"] = df["steer"].abs()

    # Rate-of-change features -- these are what actually separate "smooth,
    # controlled driving" from "jerky, reactive driving".
    df["accel"] = df["speed_kmh"].diff().fillna(0.0)
    df["brake_jerk"] = df["brake"].diff().abs().fillna(0.0)
    df["brake_energy"] = df["brake"] * df["speed_kmh"]  # braking hard while fast

    return df


def derive_safety_label(df: pd.DataFrame) -> pd.Series:
    """
    unsafe = 1 when any physically-risky pattern is present.
    This is intentionally conservative/explainable -- each rule maps to a
    concrete real-world driving hazard.
    """
    unsafe = pd.Series(False, index=df.index)

    # 1. Ground truth hazards already logged by CARLA sensors.
    unsafe |= df["collision"] == 1
    unsafe |= df["lane_invasion"] == 1

    # 2. Harsh braking at meaningful speed (loss-of-control / tailgating risk).
    unsafe |= (df["brake_jerk"] > 0.5) & (df["speed_kmh"] > 40)

    # 3. Excess steering input at speed (risk of skid / rollover / lane departure).
    unsafe |= (df["abs_steer"] > 0.5) & (df["speed_kmh"] > 60)

    # 4. Speeding in fog (visibility-limited conditions).
    unsafe |= (df["fog"] == 1) & (df["speed_kmh"] > 60)

    # 5. Hard braking in the rain (tyre skid risk on wet asphalt).
    unsafe |= (df["rain"] == 1) & (df["brake"] > 0.6) & (df["speed_kmh"] > 30)

    # 6. Aggressive acceleration/deceleration jerk in general (uncomfortable +
    #    higher accident risk).
    unsafe |= df["accel"].abs() > 25  # km/h per tick

    # 7. Automatic Emergency Braking actually fired -- by definition unsafe.
    unsafe |= df["aeb_triggered"] == 1

    # 8. Closing in on a front obstacle inside the safe braking distance
    #    without the driver actually braking (the exact scenario the
    #    predictive-braking system is meant to catch).
    unsafe |= (
        (df["obstacle_front_m"] < df["safe_braking_distance_m"])
        & (df["safe_braking_distance_m"] > 0)
        & (df["brake"] < 0.2)
        & (df["speed_kmh"] > 20)
    )

    # 9. Something (a pedestrian, a wall, another car) within a metre on
    #    either side while still moving at speed -- lane/parking risk.
    unsafe |= (df["min_side_clearance_m"] < 1.0) & (df["speed_kmh"] > 20)

    return unsafe.astype(int)


def _resolve_csv_path() -> str:
    """
    Accepts the CSV path either as a command-line argument
    (python train_safety_model.py path/to/driving_data.csv) or, if omitted,
    prompts for it interactively -- re-asking until a real file is given.
    """
    if len(sys.argv) > 1:
        return sys.argv[1]

    while True:
        path = input("Path to driving data CSV (e.g. driving_data.csv): ").strip().strip('"')
        if not path:
            print("Please enter a path (or Ctrl+C to cancel).")
            continue
        if not os.path.exists(path):
            print(f"'{path}' not found -- check the path and try again.")
            continue
        return path


def main():
    csv_path = _resolve_csv_path()
    print(f"Loading: {csv_path}")

    df = load_and_clean(csv_path)
    df = engineer_features(df)
    y = derive_safety_label(df)
    X = df[FEATURES].fillna(0.0)

    print(f"Rows: {len(df)} | Unsafe-labeled: {y.sum()} ({y.mean()*100:.1f}%)")

    if y.nunique() < 2:
        print("\nWARNING: Only one class present in the derived labels.")
        print("The model will still be saved (it will predict that single class")
        print("with high confidence) but it can't learn a real decision boundary")
        print("until your dataset contains some genuinely risky driving moments")
        print("(harsh braking, fog + speed, lane invasions, etc).")
        model = RandomForestClassifier(n_estimators=200, random_state=42)
        model.fit(X, y)
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        model = RandomForestClassifier(
            n_estimators=300, max_depth=8, min_samples_leaf=3,
            class_weight="balanced", random_state=42
        )
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]
        print("\nClassification report:")
        print(classification_report(y_test, preds, target_names=["safe", "unsafe"]))
        try:
            print(f"ROC-AUC: {roc_auc_score(y_test, probs):.3f}")
        except ValueError:
            pass

        # Refit on all data before shipping the model.
        model.fit(X, y)

    importances = sorted(
        zip(FEATURES, model.feature_importances_), key=lambda t: -t[1]
    )
    print("\nFeature importances:")
    for name, imp in importances:
        print(f"  {name:16s} {imp:.3f}")

    joblib.dump({"model": model, "features": FEATURES}, MODEL_PATH)
    print(f"\nSaved model -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
