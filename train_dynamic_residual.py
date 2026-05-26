#!/usr/bin/env python3
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

LOG_PATH = "dynamic_residual_log.csv"
MODEL_PATH = "dynamic_residual_model.json"

FEATURES = [
    "cmd0", "cmd1", "cmd2",
    "actual0", "actual1", "actual2",

    "cmd0_prev", "cmd1_prev", "cmd2_prev",
    "actual0_prev", "actual1_prev", "actual2_prev",

    "dcmd0_dt", "dcmd1_dt", "dcmd2_dt",
    "dactual0_dt", "dactual1_dt", "dactual2_dt",

    "sofa_roll", "sofa_pitch",
    "sofa_roll_prev", "sofa_pitch_prev",

    "err_roll_prev", "err_pitch_prev",
]

TARGETS = ["err_roll", "err_pitch"]


def clean_data(df):
    df = df.copy()
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # Remove unrealistic velocity spikes. Increase this if your slider jumps are intentional.
    velocity_cols = [
        "dcmd0_dt", "dcmd1_dt", "dcmd2_dt",
        "dactual0_dt", "dactual1_dt", "dactual2_dt",
    ]
    for c in velocity_cols:
        df = df[df[c].abs() < 500.0]

    # Remove first few samples because previous-state features are not meaningful yet.
    if len(df) > 10:
        df = df.iloc[10:].copy()

    return df


def train_ridge(X, Y, ridge=1e-2):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    A = X_aug.T @ X_aug
    A += ridge * np.eye(A.shape[0])
    B = X_aug.T @ Y
    return np.linalg.solve(A, B)


def predict(X, W):
    X_aug = np.hstack([X, np.ones((X.shape[0], 1))])
    return X_aug @ W


def rmse(x):
    return float(np.sqrt(np.mean(np.square(x))))


def main():
    if not os.path.exists(LOG_PATH):
        raise FileNotFoundError(f"Cannot find {LOG_PATH}. Run tendon_console_dynamic_residual.py first.")

    df = pd.read_csv(LOG_PATH)
    df = clean_data(df)

    missing = [c for c in FEATURES + TARGETS if c not in df.columns]
    if missing:
        raise RuntimeError(f"Missing columns in log file: {missing}")

    if len(df) < 80:
        raise RuntimeError(f"Not enough data: {len(df)} samples. Collect at least 500 samples if possible.")

    # Time-based split, not random split.
    split = int(len(df) * 0.7)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()

    X_train = train_df[FEATURES].to_numpy(dtype=float)
    Y_train = train_df[TARGETS].to_numpy(dtype=float)
    X_test = test_df[FEATURES].to_numpy(dtype=float)
    Y_test = test_df[TARGETS].to_numpy(dtype=float)

    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std[x_std < 1e-9] = 1.0

    X_train_n = (X_train - x_mean) / x_std
    X_test_n = (X_test - x_mean) / x_std

    W = train_ridge(X_train_n, Y_train, ridge=1e-2)
    Y_pred = predict(X_test_n, W)

    # Before correction: residual = IMU - SOFA.
    # After correction: residual_after = residual - learned_residual.
    residual_after = Y_test - Y_pred

    before_roll_rmse = rmse(Y_test[:, 0])
    before_pitch_rmse = rmse(Y_test[:, 1])
    after_roll_rmse = rmse(residual_after[:, 0])
    after_pitch_rmse = rmse(residual_after[:, 1])

    print("\n=== Physics-Informed Dynamic Residual Learning Result ===")
    print(f"Samples used: train={len(train_df)}, test={len(test_df)}")
    print(f"Roll  RMSE before: {before_roll_rmse:.3f} deg")
    print(f"Roll  RMSE after : {after_roll_rmse:.3f} deg")
    print(f"Pitch RMSE before: {before_pitch_rmse:.3f} deg")
    print(f"Pitch RMSE after : {after_pitch_rmse:.3f} deg")

    model = {
        "features": FEATURES,
        "targets": TARGETS,
        "x_mean": x_mean.tolist(),
        "x_std": x_std.tolist(),
        "W": W.tolist(),
        "definition": "residual = IMU - SOFA; corrected_SOFA = SOFA + predicted_residual",
        "before_roll_rmse_deg": before_roll_rmse,
        "before_pitch_rmse_deg": before_pitch_rmse,
        "after_roll_rmse_deg": after_roll_rmse,
        "after_pitch_rmse_deg": after_pitch_rmse,
    }

    with open(MODEL_PATH, "w") as f:
        json.dump(model, f, indent=2)
    print(f"\nSaved model to: {MODEL_PATH}")

    t = test_df["t"].to_numpy()
    t = t - t[0]

    plt.figure(figsize=(10, 5))
    plt.plot(t, Y_test[:, 0], label="Before correction: roll residual")
    plt.plot(t, residual_after[:, 0], label="After correction: roll residual")
    plt.xlabel("Time (s)")
    plt.ylabel("Roll residual (deg)")
    plt.title("Dynamic residual learning: roll error reduction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("roll_residual_before_after.png", dpi=200)

    plt.figure(figsize=(10, 5))
    plt.plot(t, Y_test[:, 1], label="Before correction: pitch residual")
    plt.plot(t, residual_after[:, 1], label="After correction: pitch residual")
    plt.xlabel("Time (s)")
    plt.ylabel("Pitch residual (deg)")
    plt.title("Dynamic residual learning: pitch error reduction")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pitch_residual_before_after.png", dpi=200)

    labels = ["Roll", "Pitch"]
    before = [before_roll_rmse, before_pitch_rmse]
    after = [after_roll_rmse, after_pitch_rmse]
    x = np.arange(len(labels))
    width = 0.35

    plt.figure(figsize=(7, 5))
    plt.bar(x - width / 2, before, width, label="Before")
    plt.bar(x + width / 2, after, width, label="After")
    plt.xticks(x, labels)
    plt.ylabel("RMSE (deg)")
    plt.title("SOFA soft-sensor residual error before/after learning")
    plt.grid(True, axis="y")
    plt.legend()
    plt.tight_layout()
    plt.savefig("residual_rmse_before_after.png", dpi=200)

    print("\nSaved figures:")
    print("  roll_residual_before_after.png")
    print("  pitch_residual_before_after.png")
    print("  residual_rmse_before_after.png")


if __name__ == "__main__":
    main()
