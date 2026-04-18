import argparse
import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from collect_data import load_data
from features import (
    build_feature_frame,
    compute_features,
    fit_scaler,
    get_feature_columns,
    get_hmm_feature_columns,
    transform_features,
)
from postprocess import smooth_regime_sequence
from postprocess import apply_hmm_consolidation_override, build_hmm_consolidation_rules
from analysis import (
    compute_regime_feature_profile,
    print_regime_explainability_report,
    plot_regime_feature_heatmap,
    rolling_window_validation,
    analyze_rolling_results,
)

try:
    from hmmlearn.hmm import GaussianHMM
except ImportError:
    GaussianHMM = None


REGIME_LABELS = {
    0: "Consolidation",
    1: "Bull",
    2: "Bear",
}

REGIME_PLOT_LABELS = {
    0: "Consolidation",
    1: "Bull",
    2: "Bear",
}

REGIME_COLORS = {
    0: "#C7CCD3",
    1: "#0B3D20",
    2: "#6E1F1B",
}

REGIME_NAME_COLORS = {REGIME_PLOT_LABELS[key]: REGIME_COLORS[key] for key in REGIME_COLORS}


def load_and_prepare_data(model_type="hmm", test_size=0.2, scaler_type="standard"):
    """Load data, build features, and split by time."""
    df = load_data()
    if df is None:
        raise FileNotFoundError("Please run collect_data.py first.")

    df = compute_features(df)
    feature_cols = get_hmm_feature_columns() if model_type == "hmm" else get_feature_columns()
    df_model, feature_df = build_feature_frame(df, feature_cols)

    min_required = 100
    if len(df_model) < min_required:
        raise ValueError(f"Need at least {min_required} model-ready rows, got {len(df_model)}")

    split_idx = int(len(df_model) * (1 - test_size))
    if split_idx < min_required:
        split_idx = min_required
    if split_idx > len(df_model) - min_required:
        split_idx = len(df_model) - min_required

    train_df = df_model.iloc[:split_idx].copy()
    test_df = df_model.iloc[split_idx:].copy()

    train_features = feature_df.iloc[:split_idx].copy()
    test_features = feature_df.iloc[split_idx:].copy()

    X_train, scaler = fit_scaler(train_features, scaler_type=scaler_type)
    X_test = transform_features(test_features, scaler)
    X_all = transform_features(feature_df, scaler)

    print(f"\nModel type: {model_type}")
    print(f"Raw rows: {len(df)}")
    print(f"Model-ready rows: {len(df_model)}")
    print(f"Feature count: {len(feature_cols)}")
    print(f"Train set: {len(train_df)} rows ({train_df['datetime'].iloc[0]} ~ {train_df['datetime'].iloc[-1]})")
    print(f"Test set: {len(test_df)} rows ({test_df['datetime'].iloc[0]} ~ {test_df['datetime'].iloc[-1]})")

    return {
        "df_model": df_model,
        "feature_cols": feature_cols,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "X_all": X_all,
        "scaler": scaler,
    }


def train_kmeans(X_train, n_states=3, random_state=42):
    if len(X_train) < n_states * 10:
        raise ValueError(f"Need at least {n_states * 10} samples for KMeans, got {len(X_train)}")
    model = KMeans(n_clusters=n_states, random_state=random_state, n_init=10)
    model.fit(X_train)
    return model


def train_hmm(X_train, n_states=3, random_state=42, n_init=3, n_iter=500):
    if GaussianHMM is None:
        raise ImportError("Please install hmmlearn to use the HMM model.")
    min_required = n_states * 10
    if len(X_train) < min_required:
        raise ValueError(f"Need at least {min_required} samples for HMM, got {len(X_train)}")

    best_model = None
    best_score = -np.inf

    for seed_offset in range(n_init):
        candidate = GaussianHMM(
            n_components=n_states,
            covariance_type="diag",
            n_iter=n_iter,
            random_state=random_state + seed_offset,
            min_covar=1e-4,
        )
        candidate.fit(X_train)
        score = candidate.score(X_train)
        if score > best_score:
            best_score = score
            best_model = candidate

    return best_model


def predict_states(model, X, model_type):
    if model_type == "hmm":
        return model.predict(X)
    return model.predict(X)


def predict_state_confidence(model, X, model_type):
    if model_type == "hmm":
        return model.predict_proba(X)

    distances = model.transform(X)
    inverse_distance = 1 / (distances + 1e-9)
    return inverse_distance / inverse_distance.sum(axis=1, keepdims=True)


def infer_regime_mapping(train_state_df, state_col="state"):
    """Map each state to Bull, Bear, or Consolidation using multi-dimensional analysis.

    Breaks from single-score approach. Uses three normalized dimensions:
    - trend_score:    avg_return_30d + avg_return_90d  (higher = bullish)
    - volatility_score: avg_volatility_30d              (higher = more volatile)
    - direction_score: adx_direction                   (higher = more directional)
    - consolidation_score: avg_bb_mid_distance + avg_rsi_mid_distance
                         + avg_direction_flip_rate

    Each dimension is min-max normalized across states, then combined into
    a composite score for bull/bear ranking. The remaining state is labeled
    as Consolidation regardless of its scores.
    """
    state_summary = (
        train_state_df.groupby(state_col)
        .agg(
            samples=(state_col, "size"),
            avg_return_30d=("returns_30d", "mean"),
            avg_return_90d=("returns_90d", "mean"),
            avg_volatility_30d=("volatility_30d", "mean"),
            avg_adx=("adx", "mean"),
            avg_rsi_14=("rsi_14", "mean"),
            avg_bb_mid_distance_abs=("bb_mid_distance_abs", "mean"),
            avg_range_20=("range_20_ratio", "mean"),
            avg_direction_flip_rate=("direction_flip_rate_20", "mean"),
            avg_adx_direction=("adx_direction", "mean"),
            avg_macd_hist_ratio=("macd_hist_ratio", "mean"),
        )
        .sort_index()
    )

    def _norm(series):
        mn, mx = series.min(), series.max()
        if mx == mn:
            return series * 0 + 0.5
        return (series - mn) / (mx - mn)

    state_summary["trend_score"] = (
        state_summary["avg_return_30d"] * 0.4 + state_summary["avg_return_90d"] * 0.6
    )
    state_summary["trend_norm"] = _norm(state_summary["trend_score"])
    state_summary["volatility_norm"] = _norm(state_summary["avg_volatility_30d"])
    state_summary["direction_norm"] = _norm(state_summary["avg_adx_direction"])

    state_summary["consolidation_score"] = (
        state_summary["avg_bb_mid_distance_abs"] * 2.0
        + (state_summary["avg_rsi_14"] - 0.5).abs() * 1.5
        + state_summary["avg_direction_flip_rate"] * 1.0
        + state_summary["avg_range_20"] * 1.0
    )
    state_summary["consolidation_norm"] = _norm(state_summary["consolidation_score"])

    state_summary["bull_composite"] = (
        state_summary["trend_norm"] * 0.55
        + (1 - state_summary["volatility_norm"]) * 0.25
        + state_summary["direction_norm"] * 0.20
    )
    state_summary["bear_composite"] = (
        (1 - state_summary["trend_norm"]) * 0.55
        + state_summary["volatility_norm"] * 0.25
        + (1 - state_summary["direction_norm"]) * 0.20
    )

    bull_state = state_summary["bull_composite"].idxmax()
    bear_state = state_summary["bear_composite"].idxmax()

    if bull_state == bear_state:
        scores = state_summary["bull_composite"]
        sorted_states = scores.sort_values(ascending=False).index.tolist()
        bull_state = sorted_states[0]
        bear_state = sorted_states[1] if len(sorted_states) > 1 else sorted_states[0]

    remaining_states = list(set(state_summary.index) - {bull_state, bear_state})

    state_to_regime = {int(bull_state): 1, int(bear_state): 2}
    for state in remaining_states:
        state_to_regime[int(state)] = 0

    state_summary["regime_code"] = state_summary.index.map(state_to_regime)
    state_summary["regime_name"] = state_summary["regime_code"].map(REGIME_LABELS)
    state_summary["regime_plot_name"] = state_summary["regime_code"].map(REGIME_PLOT_LABELS)
    return state_to_regime, state_summary


def apply_regime_mapping(df, state_to_regime, state_col="state"):
    result = df.copy()
    result["regime"] = result[state_col].map(state_to_regime).astype(int)
    result["regime_name"] = result["regime"].map(REGIME_LABELS)
    result["regime_plot_name"] = result["regime"].map(REGIME_PLOT_LABELS)
    return result


def smooth_regime_assignments(df, model_type, min_run_length):
    """Smooth only KMeans regime assignments to reduce short-lived flips."""
    result = df.copy()
    if model_type != "kmeans" or min_run_length <= 1:
        return result

    result["regime_raw"] = result["regime"]
    result["regime"] = smooth_regime_sequence(result["regime"].to_numpy(), min_run_length=min_run_length)
    result["regime_name"] = result["regime"].map(REGIME_LABELS)
    result["regime_plot_name"] = result["regime"].map(REGIME_PLOT_LABELS)
    return result


def apply_hmm_postprocess(df, regime_confidence, hmm_rules):
    """Apply consolidation-biased HMM override and convert to final regimes."""
    adjusted_confidence, override_mask, bear_priority_mask, bull_priority_mask = apply_hmm_consolidation_override(
        df, regime_confidence, hmm_rules
    )
    result = df.copy()
    result["regime_confidence_consolidation"] = adjusted_confidence[:, 0]
    result["regime_confidence_bull"] = adjusted_confidence[:, 1]
    result["regime_confidence_bear"] = adjusted_confidence[:, 2]
    result["consolidation_override"] = override_mask.astype(int)
    result["bear_priority_override"] = bear_priority_mask.astype(int)
    result["bull_priority_override"] = bull_priority_mask.astype(int)
    result["regime"] = adjusted_confidence.argmax(axis=1).astype(int)
    result["regime_name"] = result["regime"].map(REGIME_LABELS)
    result["regime_plot_name"] = result["regime"].map(REGIME_PLOT_LABELS)
    return result


def evaluate_model(model, model_type, X_train, X_test, train_labeled_df, test_labeled_df):
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)

    if model_type == "hmm":
        train_metric = model.score(X_train) / len(X_train)
        test_metric = model.score(X_test) / len(X_test)
        print(f"\nAverage log-likelihood per sample (train): {train_metric:.4f}")
        print(f"Average log-likelihood per sample (test):  {test_metric:.4f}")
        metrics = {
            "train_avg_log_likelihood": float(train_metric),
            "test_avg_log_likelihood": float(test_metric),
        }
    else:
        train_states = model.predict(X_train)
        test_states = model.predict(X_test)
        train_metric = silhouette_score(X_train, train_states)
        test_metric = silhouette_score(X_test, test_states)
        print(f"\nSilhouette score (train): {train_metric:.4f}")
        print(f"Silhouette score (test):  {test_metric:.4f}")
        metrics = {
            "train_silhouette": float(train_metric),
            "test_silhouette": float(test_metric),
        }

    for split_name, split_df in (("Train", train_labeled_df), ("Test", test_labeled_df)):
        print(f"\n{split_name} regime distribution:")
        counts = split_df["regime"].value_counts(normalize=True).sort_index()
        for regime_code, share in counts.items():
            print(f"  {REGIME_LABELS[regime_code]}: {share * 100:.1f}%")

    return metrics


def plot_state_summary(state_summary, output_path="results/state_summary.png"):
    plot_df = state_summary.reset_index(names="state")
    plt.figure(figsize=(12, 4.0))
    size_scale = 800
    sns.scatterplot(
        data=plot_df,
        x="avg_return_90d",
        y="avg_volatility_30d",
        hue="regime_plot_name",
        palette=REGIME_NAME_COLORS,
        size="samples",
        sizes=(180, size_scale),
        legend=False,
        edgecolor="white",
        linewidth=1.0,
    )

    for _, row in plot_df.iterrows():
        plt.text(
            row["avg_return_90d"],
            row["avg_volatility_30d"],
            f"{row['regime_plot_name']}\nN={int(row['samples'])}",
            fontsize=10,
            fontweight="bold",
            ha="left",
            va="bottom",
        )

    plt.title("State Summary: 90D Return vs 30D Volatility")
    plt.xlabel("Average 90D Return")
    plt.ylabel("Average 30D Volatility")
    legend_handles = [
        plt.scatter([], [], s=220, color=REGIME_COLORS[code], label=REGIME_PLOT_LABELS[code], edgecolors="white", linewidths=1.0)
        for code in REGIME_PLOT_LABELS
    ]
    plt.legend(handles=legend_handles, title="Regime", loc="upper right")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved state summary chart to {output_path}")


def plot_regime_distribution(df):
    counts = df["regime"].value_counts().sort_index()
    labels = [REGIME_PLOT_LABELS[code] for code in counts.index]
    colors = [REGIME_COLORS[code] for code in counts.index]
    plt.figure(figsize=(9, 3.0))
    plt.bar(labels, counts.values, color=colors)
    plt.xticks(rotation=10)
    plt.ylabel("Days")
    plt.title("Historical BTC Market Regime Distribution")
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/regime_distribution.png", dpi=150)
    plt.close()
    print("Saved regime distribution chart to results/regime_distribution.png")


def save_model(
    model,
    scaler,
    feature_cols,
    model_type,
    state_to_regime,
    state_summary,
    metrics,
    min_regime_run,
    hmm_rules,
    regime_profile=None,
    path="models",
):
    os.makedirs(path, exist_ok=True)

    joblib.dump(model, f"{path}/model.pkl")
    joblib.dump(scaler, f"{path}/scaler.pkl")

    with open(f"{path}/feature_cols.txt", "w", encoding="utf-8") as file:
        file.write("\n".join(feature_cols))

    metadata = {
        "model_type": model_type,
        "state_to_regime": {str(key): value for key, value in state_to_regime.items()},
        "regime_labels": {str(key): value for key, value in REGIME_LABELS.items()},
        "metrics": metrics,
        "state_summary": state_summary.reset_index(names="state").to_dict(orient="records"),
        "postprocess": {
            "kmeans_min_regime_run": min_regime_run if model_type == "kmeans" else 1,
            "hmm_consolidation_rules": hmm_rules if model_type == "hmm" else {},
        },
    }
    if regime_profile:
        serializable_profile = {}
        for regime_code, info in regime_profile.items():
            serializable_profile[str(regime_code)] = {
                "mean": info["mean"],
                "effect_size": info["effect_size"],
                "top_positive": info["top_positive"],
                "top_negative": info["top_negative"],
            }
        metadata["regime_feature_profile"] = serializable_profile
    if model_type == "hmm":
        metadata["transition_matrix"] = model.transmat_.tolist()
        metadata["start_probability"] = model.startprob_.tolist()

    with open(f"{path}/metadata.json", "w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)

    print(f"\nSaved model artifacts to {path}/")


def main():
    parser = argparse.ArgumentParser(description="Unsupervised BTC market regime training")
    parser.add_argument("--model", type=str, default="hmm", choices=["hmm", "kmeans"], help="Model type")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test set ratio")
    parser.add_argument(
        "--scaler",
        type=str,
        default="standard",
        choices=["standard", "minmax"],
        help="Feature scaling method",
    )
    parser.add_argument("--states", type=int, default=3, help="Number of latent states")
    parser.add_argument(
        "--min-regime-run",
        type=int,
        default=6,
        help="Minimum consecutive candles for KMeans regime smoothing",
    )
    parser.add_argument(
        "--rolling-val",
        action="store_true",
        help="Run rolling window validation after training",
    )
    parser.add_argument(
        "--rolling-train-window",
        type=int,
        default=1000,
        help="Training window size for rolling validation (candles)",
    )
    parser.add_argument(
        "--rolling-val-window",
        type=int,
        default=100,
        help="Validation window size for rolling validation (candles)",
    )
    parser.add_argument(
        "--rolling-step",
        type=int,
        default=50,
        help="Step size between rolling windows (candles)",
    )
    args = parser.parse_args()

    data_bundle = load_and_prepare_data(
        model_type=args.model,
        test_size=args.test_size,
        scaler_type=args.scaler,
    )

    if args.model == "hmm":
        model = train_hmm(data_bundle["X_train"], n_states=args.states)
    else:
        model = train_kmeans(data_bundle["X_train"], n_states=args.states)

    train_labeled_df = data_bundle["train_df"].copy()
    test_labeled_df = data_bundle["test_df"].copy()
    all_labeled_df = data_bundle["df_model"].copy()

    train_labeled_df["state"] = predict_states(model, data_bundle["X_train"], args.model)
    test_labeled_df["state"] = predict_states(model, data_bundle["X_test"], args.model)
    all_labeled_df["state"] = predict_states(model, data_bundle["X_all"], args.model)

    state_to_regime, state_summary = infer_regime_mapping(train_labeled_df)

    train_labeled_df = apply_regime_mapping(train_labeled_df, state_to_regime)
    test_labeled_df = apply_regime_mapping(test_labeled_df, state_to_regime)
    all_labeled_df = apply_regime_mapping(all_labeled_df, state_to_regime)

    hmm_rules = {}
    if args.model == "hmm":
        train_confidence = predict_state_confidence(model, data_bundle["X_train"], args.model)
        test_confidence = predict_state_confidence(model, data_bundle["X_test"], args.model)
        all_confidence = predict_state_confidence(model, data_bundle["X_all"], args.model)

        train_regime_confidence = np.zeros((len(train_labeled_df), len(REGIME_LABELS)))
        test_regime_confidence = np.zeros((len(test_labeled_df), len(REGIME_LABELS)))
        all_regime_confidence = np.zeros((len(all_labeled_df), len(REGIME_LABELS)))
        for state_id, regime_code in state_to_regime.items():
            train_regime_confidence[:, regime_code] += train_confidence[:, state_id]
            test_regime_confidence[:, regime_code] += test_confidence[:, state_id]
            all_regime_confidence[:, regime_code] += all_confidence[:, state_id]

        # Learn postprocess thresholds from the training split only.
        hmm_rules = build_hmm_consolidation_rules(train_labeled_df)
        train_labeled_df = apply_hmm_postprocess(train_labeled_df, train_regime_confidence, hmm_rules)
        test_labeled_df = apply_hmm_postprocess(test_labeled_df, test_regime_confidence, hmm_rules)
        all_labeled_df = apply_hmm_postprocess(all_labeled_df, all_regime_confidence, hmm_rules)

    train_labeled_df = smooth_regime_assignments(train_labeled_df, args.model, args.min_regime_run)
    test_labeled_df = smooth_regime_assignments(test_labeled_df, args.model, args.min_regime_run)
    all_labeled_df = smooth_regime_assignments(all_labeled_df, args.model, args.min_regime_run)

    metrics = evaluate_model(
        model,
        args.model,
        data_bundle["X_train"],
        data_bundle["X_test"],
        train_labeled_df,
        test_labeled_df,
    )

    print("\nLatent state summary:")
    printable_summary = state_summary.copy()
    printable_summary["avg_adx"] = printable_summary["avg_adx"] * 100
    printable_summary["avg_rsi_14"] = printable_summary["avg_rsi_14"] * 100
    print(printable_summary.round(4).to_string())

    if args.model == "hmm":
        print("\nHMM transition matrix:")
        print(np.round(model.transmat_, 4))
    else:
        print(f"\nKMeans regime smoothing: minimum run length = {args.min_regime_run} candles")

    plot_state_summary(state_summary)
    plot_regime_distribution(all_labeled_df)

    feature_cols = data_bundle["feature_cols"]

    profile, profile_stats = compute_regime_feature_profile(
        all_labeled_df, feature_cols, regime_col="regime"
    )
    print_regime_explainability_report(profile, REGIME_LABELS)
    plot_regime_feature_heatmap(profile, REGIME_LABELS, feature_cols)

    save_model(
        model,
        data_bundle["scaler"],
        data_bundle["feature_cols"],
        args.model,
        state_to_regime,
        state_summary,
        metrics,
        args.min_regime_run,
        hmm_rules,
        regime_profile=profile,
    )

    if args.rolling_val:
        print("\n" + "=" * 60)
        print("Running Rolling Window Validation")
        print("=" * 60)

        if args.model == "hmm":
            model_class = GaussianHMM
            model_params = {
                "n_components": args.states,
                "covariance_type": "diag",
                "n_iter": 500,
                "random_state": 42,
                "min_covar": 1e-4,
            }
        else:
            model_class = KMeans
            model_params = {
                "n_clusters": args.states,
                "random_state": 42,
                "n_init": 5,
            }

        rolling_results = rolling_window_validation(
            df=all_labeled_df,
            feature_cols=feature_cols,
            model_class=model_class,
            model_params=model_params,
            scaler_type=args.scaler,
            train_window=args.rolling_train_window,
            val_window=args.rolling_val_window,
            step=args.rolling_step,
        )
        analyze_rolling_results(rolling_results)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
