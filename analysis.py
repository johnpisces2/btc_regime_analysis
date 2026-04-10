import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os


def compute_regime_feature_profile(df, feature_cols, regime_col="regime"):
    stats = (
        df.groupby(regime_col)[feature_cols]
        .agg(["mean", "std"])
        .round(6)
    )
    overall_mean = df[feature_cols].mean()
    overall_std = df[feature_cols].std()

    profile = {}
    for regime in df[regime_col].unique():
        regime_data = df.loc[df[regime_col] == regime, feature_cols]
        regime_mean = regime_data.mean()

        effect_size = {}
        for col in feature_cols:
            se = overall_std[col] + 1e-10
            effect_size[col] = float((regime_mean[col] - overall_mean[col]) / se)

        sorted_features = sorted(effect_size.items(), key=lambda x: abs(x[1]), reverse=True)
        profile[int(regime)] = {
            "mean": regime_mean.to_dict(),
            "effect_size": effect_size,
            "top_positive": sorted_features[:5],
            "top_negative": [(k, v) for k, v in sorted_features if v < 0][:5],
        }
    return profile, stats


def print_regime_explainability_report(profile, labels, output_dir="results"):
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Regime Feature Explainability Report")
    print("=" * 80)

    for regime_code, regime_name in labels.items():
        if regime_code not in profile:
            continue
        info = profile[regime_code]

        print(f"\n--- Regime {regime_code}: {regime_name} ---")
        print("Top distinguishing features (effect size vs overall mean):")

        top_features = sorted(
            info["effect_size"].items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:8]

        for feat, effect in top_features:
            direction = "↑" if effect > 0 else "↓"
            print(f"  {direction} {feat:30s}: {effect:+.3f}")

        if info["top_positive"]:
            print("  Positive deviation features:")
            for feat, val in info["top_positive"][:3]:
                print(f"    + {feat}: {val:+.4f}")
        if info["top_negative"]:
            print("  Negative deviation features:")
            for feat, val in info["top_negative"][:3]:
                print(f"    - {feat}: {val:+.4f}")


def plot_regime_feature_heatmap(profile, labels, feature_cols, output_path="results/regime_feature_profile.png"):
    regime_codes = sorted(profile.keys())
    n_features = len(feature_cols)
    n_regimes = len(regime_codes)

    effect_matrix = np.zeros((n_regimes, n_features))
    for i, regime_code in enumerate(regime_codes):
        for j, feat in enumerate(feature_cols):
            effect_matrix[i, j] = profile[regime_code]["effect_size"].get(feat, 0)

    fig, ax = plt.subplots(figsize=(max(14, n_features * 0.4), max(4, n_regimes * 2)))

    labels_list = [labels[rc] for rc in regime_codes]
    sns.heatmap(
        effect_matrix,
        xticklabels=feature_cols,
        yticklabels=labels_list,
        cmap="RdYlGn",
        center=0,
        annot=False,
        fmt=".2f",
        ax=ax,
        cbar_kws={"label": "Effect Size (z-score)"},
    )

    ax.set_title("Regime Feature Profile: Deviation from Overall Mean", fontsize=13, fontweight="bold")
    ax.set_xlabel("Features")
    ax.set_ylabel("Regime")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved regime feature profile to {output_path}")


def rolling_window_validation(
    df,
    feature_cols,
    model_class,
    model_params,
    scaler_type="standard",
    train_window=1000,
    val_window=100,
    step=50,
    min_train_samples=500,
):
    results = []

    total_windows = len(range(min_train_samples, len(df) - val_window, step))
    print(f"\nRolling Window Validation: {total_windows} windows")

    for i, train_end in enumerate(range(min_train_samples, len(df) - val_window, step)):
        train_start = max(0, train_end - train_window)
        val_start = train_end
        val_end = min(val_start + val_window, len(df))

        train_slice = df.iloc[train_start:train_end].copy()
        val_slice = df.iloc[val_start:val_end].copy()

        if len(train_slice) < min_train_samples:
            continue

        feature_df_train = train_slice[feature_cols].replace([np.inf, -np.inf], np.nan)
        valid_train = feature_df_train.notna().all(axis=1)
        train_clean = train_slice.loc[valid_train]
        train_feat_clean = feature_df_train.loc[valid_train]

        if len(train_clean) < min_train_samples:
            continue

        from sklearn.preprocessing import MinMaxScaler, StandardScaler
        if scaler_type == "minmax":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        X_train = scaler.fit_transform(train_feat_clean)

        try:
            model = model_class(**model_params)
            model.fit(X_train)

            feature_df_val = val_slice[feature_cols].replace([np.inf, -np.inf], np.nan)
            valid_val = feature_df_val.notna().all(axis=1)
            val_clean = val_slice.loc[valid_val]
            val_feat_clean = feature_df_val.loc[valid_val]

            if len(val_clean) == 0:
                continue

            X_val = scaler.transform(val_feat_clean)

            if hasattr(model, "predict_proba"):
                val_confidence = model.predict_proba(X_val)
                val_states = model.predict(X_val)
            else:
                val_states = model.predict(X_val)
                distances = model.transform(X_val)
                inverse_distance = 1 / (distances + 1e-9)
                val_confidence = inverse_distance / inverse_distance.sum(axis=1, keepdims=True)

            val_clean = val_clean.copy()
            val_clean["state"] = val_states

            regime_stats = _infer_regime_mapping_fast(val_clean)

            val_confidence_aligned = np.zeros((len(val_clean), 3))
            n_model_states = val_confidence.shape[1]
            for state_id, regime_code in regime_stats.items():
                if regime_code in range(3) and state_id in range(n_model_states):
                    val_confidence_aligned[:, regime_code] += val_confidence[:, state_id]

            regime_probs = val_confidence_aligned.mean(axis=0)
            regime_labels = ["Consolidation", "Bull", "Bear"]
            window_predicted = {regime_labels[i]: regime_probs[i] for i in range(3)}

            regime_changes = np.sum(np.diff(val_clean["state"].values) != 0)

            window_result = {
                "window": i,
                "train_start": train_start,
                "train_end": train_end,
                "val_start": val_start,
                "val_end": val_end,
                "n_train": len(train_clean),
                "n_val": len(val_clean),
                "prob_consolidation": regime_probs[0],
                "prob_bull": regime_probs[1],
                "prob_bear": regime_probs[2],
                "entropy": _entropy(regime_probs),
                "regime_changes": regime_changes,
                "regime_changes_per_candle": regime_changes / max(1, len(val_clean)),
                "dominant_regime": regime_labels[int(np.argmax(regime_probs))],
                "regime_stability": 1.0 - regime_changes / max(1, len(val_clean)),
            }
            results.append(window_result)

            if (i + 1) % 20 == 0 or i == 0:
                print(f"  [{i+1}/{total_windows}] window {i}: dominant={window_result['dominant_regime']}, "
                      f"stability={window_result['regime_stability']:.2f}")

        except Exception as e:
            print(f"  [WARN] window {i} failed: {e}")
            continue

    return pd.DataFrame(results)


def _entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    return -np.sum(p * np.log(p))


def _infer_regime_mapping_fast(df, state_col="state"):
    required_cols = ["returns_30d", "returns_90d"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        return {int(s): i % 3 for i, s in enumerate(df[state_col].unique())}

    state_summary = (
        df.groupby(state_col)
        .agg(
            avg_return_30d=("returns_30d", "mean"),
            avg_return_90d=("returns_90d", "mean"),
        )
        .reset_index()
    )

    state_summary["trend_score"] = (
        state_summary["avg_return_30d"].fillna(0) * 0.4
        + state_summary["avg_return_90d"].fillna(0) * 0.6
    )

    sorted_states = state_summary.sort_values("trend_score")
    state_list = sorted_states[state_col].tolist()
    n_states = len(state_list)

    if n_states == 1:
        return {int(state_list[0]): 0}
    elif n_states == 2:
        bull_idx = 1 if state_summary.loc[state_summary[state_col] == state_list[1], "trend_score"].values[0] > \
                           state_summary.loc[state_summary[state_col] == state_list[0], "trend_score"].values[0] else 0
        bull_state = state_list[bull_idx]
        bear_state = state_list[1 - bull_idx]
        return {int(bull_state): 1, int(bear_state): 2}
    else:
        state_to_regime = {}
        for idx, state in enumerate(state_list):
            if idx == 0:
                state_to_regime[int(state)] = 0
            elif idx == 1:
                state_to_regime[int(state)] = 1
            else:
                state_to_regime[int(state)] = 2
        return state_to_regime


def analyze_rolling_results(results_df, output_dir="results"):
    if results_df.empty:
        print("No rolling window results to analyze.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 80)
    print("Rolling Window Validation Summary")
    print("=" * 80)

    print(f"\nTotal windows evaluated: {len(results_df)}")
    print(f"Train window size: median = {results_df['n_train'].median():.0f}")
    print(f"Val window size: median = {results_df['n_val'].median():.0f}")

    print("\n--- Regime Distribution Across Windows ---")
    for regime in ["Consolidation", "Bull", "Bear"]:
        col = f"prob_{regime.lower()}"
        print(f"  {regime:20s}: mean={results_df[col].mean():.3f}, "
              f"std={results_df[col].std():.3f}, "
              f"min={results_df[col].min():.3f}, max={results_df[col].max():.3f}")

    print("\n--- Regime Stability ---")
    print(f"  Avg regime changes per window: {results_df['regime_changes'].mean():.2f}")
    print(f"  Avg regime changes per candle: {results_df['regime_changes_per_candle'].mean():.3f}")
    print(f"  Avg regime stability score: {results_df['regime_stability'].mean():.3f}")

    print("\n--- Entropy (Uncertainty) ---")
    print(f"  Mean entropy: {results_df['entropy'].mean():.3f}")
    print(f"  High uncertainty windows (entropy > 0.9): "
          f"{(results_df['entropy'] > 0.9).sum()} / {len(results_df)}")

    dominant_counts = results_df["dominant_regime"].value_counts()
    print("\n--- Dominant Regime Frequency ---")
    for regime, count in dominant_counts.items():
        print(f"  {regime}: {count} windows ({count/len(results_df)*100:.1f}%)")

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    regime_cols = ["prob_consolidation", "prob_bull", "prob_bear"]
    for col, label in zip(regime_cols, ["Consolidation", "Bull", "Bear"]):
        ax.plot(results_df["window"], results_df[col], label=label, linewidth=1.5)
    ax.set_title("Regime Probabilities Over Rolling Windows")
    ax.set_xlabel("Window")
    ax.set_ylabel("Probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(results_df["window"], results_df["regime_stability"], color="steelblue", linewidth=1.5)
    ax.axhline(results_df["regime_stability"].mean(), color="red", linestyle="--", label="Mean")
    ax.set_title("Regime Stability Score Over Windows")
    ax.set_xlabel("Window")
    ax.set_ylabel("Stability (1 = perfect)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    ax.plot(results_df["window"], results_df["entropy"], color="purple", linewidth=1.5)
    ax.axhline(0.85, color="orange", linestyle="--", label="Entropy threshold (0.85)")
    ax.set_title("Entropy (Model Uncertainty) Over Windows")
    ax.set_xlabel("Window")
    ax.set_ylabel("Entropy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    results_df["dominant_regime"].value_counts().plot(
        kind="bar", ax=ax, color=["gray", "darkgreen", "darkred"]
    )
    ax.set_title("Dominant Regime Frequency")
    ax.set_ylabel("Window Count")
    ax.tick_params(axis="x", rotation=0)

    plt.tight_layout()
    output_path = os.path.join(output_dir, "rolling_validation.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\nSaved rolling validation chart to {output_path}")

    summary_path = os.path.join(output_dir, "rolling_validation_summary.csv")
    results_df.to_csv(summary_path, index=False)
    print(f"Saved rolling validation data to {summary_path}")

    return results_df
