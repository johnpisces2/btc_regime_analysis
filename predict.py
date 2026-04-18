import argparse
import json
import os

import joblib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Patch

from collect_data import fetch_btc_ohlcv, load_data
from features import build_feature_frame, compute_features, transform_features
from postprocess import apply_hmm_consolidation_override, smooth_regime_sequence


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

REGIME_ALPHA = {
    0: 0.10,
    1: 0.06,
    2: 0.06,
}

EMPTY_RESULT_MESSAGE = (
    "No model-ready rows available for prediction. "
    "Check the input data range, symbol, exchange, or timeframe."
)


def positive_int(value):
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def has_prediction_rows(result):
    return result is not None and not result.empty


def load_model(path="models"):
    model = joblib.load(f"{path}/model.pkl")
    scaler = joblib.load(f"{path}/scaler.pkl")

    with open(f"{path}/feature_cols.txt", "r", encoding="utf-8") as file:
        feature_cols = [line.strip() for line in file.readlines() if line.strip()]

    with open(f"{path}/metadata.json", "r", encoding="utf-8") as file:
        metadata = json.load(file)

    mapping_key = "state_to_regime" if "state_to_regime" in metadata else "cluster_to_regime"
    state_to_regime = {int(key): value for key, value in metadata[mapping_key].items()}
    model_type = metadata.get("model_type", "kmeans")
    print(f"Loaded model from {path}/ ({model_type})")
    return model, scaler, feature_cols, model_type, state_to_regime, metadata


def predict_states(model, X, model_type):
    return model.predict(X)


def predict_state_confidence(model, X, model_type):
    if model_type == "hmm":
        return model.predict_proba(X)

    distances = model.transform(X)
    inverse_distance = 1 / (distances + 1e-9)
    return inverse_distance / inverse_distance.sum(axis=1, keepdims=True)


def predict(df, model, scaler, feature_cols, model_type, state_to_regime, metadata=None):
    feature_ready_df = compute_features(df)
    feature_ready_df, feature_df = build_feature_frame(feature_ready_df, feature_cols)

    if len(feature_df) == 0:
        result = feature_ready_df.iloc[:0].copy()
        empty_arr = np.array([], dtype=int)
        result["state"] = empty_arr
        result["prediction_raw"] = empty_arr
        result["prediction"] = empty_arr
        result["prediction_name"] = np.array([], dtype=object)
        result["prob_consolidation"] = np.array([], dtype=float)
        result["prob_bull"] = np.array([], dtype=float)
        result["prob_bear"] = np.array([], dtype=float)
        return result

    X = transform_features(feature_df, scaler)
    states = predict_states(model, X, model_type)
    state_confidence = predict_state_confidence(model, X, model_type)

    result = feature_ready_df.copy()
    result["state"] = states
    result["prediction_raw"] = result["state"].map(state_to_regime).astype(int)
    result["prediction"] = result["prediction_raw"]

    if model_type == "kmeans":
        postprocess = metadata.get("postprocess", {}) if metadata else {}
        min_regime_run = int(postprocess.get("kmeans_min_regime_run", 1))
        if min_regime_run > 1:
            result["prediction"] = smooth_regime_sequence(
                result["prediction"].to_numpy(),
                min_run_length=min_regime_run,
            )

    result["prediction_name"] = result["prediction"].map(REGIME_LABELS)

    regime_confidence = np.zeros((len(result), len(REGIME_LABELS)))
    for state_id, regime_code in state_to_regime.items():
        regime_confidence[:, regime_code] += state_confidence[:, state_id]

    if model_type == "hmm":
        postprocess = metadata.get("postprocess", {}) if metadata else {}
        hmm_rules = postprocess.get("hmm_consolidation_rules", {})
        regime_confidence, override_mask, bear_priority_mask, bull_priority_mask = apply_hmm_consolidation_override(
            result, regime_confidence, hmm_rules
        )
        result["consolidation_override"] = override_mask.astype(int)
        result["bear_priority_override"] = bear_priority_mask.astype(int)
        result["bull_priority_override"] = bull_priority_mask.astype(int)

    result["prob_consolidation"] = regime_confidence[:, 0]
    result["prob_bull"] = regime_confidence[:, 1]
    result["prob_bear"] = regime_confidence[:, 2]
    result["prediction"] = regime_confidence.argmax(axis=1).astype(int)
    result["prediction_name"] = result["prediction"].map(REGIME_LABELS)
    return result


def _iter_regime_segments(plot_df):
    """Yield contiguous regime segments for background shading."""
    if plot_df.empty:
        return

    start_idx = 0
    predictions = plot_df["prediction"].to_numpy()
    datetimes = plot_df["datetime"].to_numpy()

    for idx in range(1, len(plot_df)):
        if predictions[idx] != predictions[start_idx]:
            yield predictions[start_idx], datetimes[start_idx], datetimes[idx]
            start_idx = idx

    if len(plot_df) == 1:
        end_time = datetimes[0]
    else:
        end_time = datetimes[-1]
    yield predictions[start_idx], datetimes[start_idx], end_time


def get_current_status(symbol="BTC/USDT", exchange_id="binance", timeframe="4h"):
    if not os.path.exists("models/model.pkl"):
        print("Model not found. Please run train.py first.")
        return None

    model, scaler, feature_cols, model_type, state_to_regime, metadata = load_model()

    print(f"\nFetching latest {symbol} {timeframe} data...")
    df = fetch_btc_ohlcv(
        symbol=symbol,
        exchange_id=exchange_id,
        start_date="2018-01-01",
        timeframe=timeframe,
    )

    result = predict(df, model, scaler, feature_cols, model_type, state_to_regime, metadata)
    if not has_prediction_rows(result):
        print(EMPTY_RESULT_MESSAGE)
        return result

    latest = result.iloc[-1]

    print("\n" + "=" * 60)
    print(f"Current market state ({latest['datetime']})")
    print("=" * 60)
    print(f"BTC price: ${latest['close']:,.2f}")
    print(f"Latent state: state {int(latest['state'])}")
    print(f"Market regime: {REGIME_LABELS[latest['prediction']]}")
    print("\nPosterior probabilities:")
    print(f"  Consolidation: {latest['prob_consolidation'] * 100:.1f}%")
    print(f"  Bull: {latest['prob_bull'] * 100:.1f}%")
    print(f"  Bear: {latest['prob_bear'] * 100:.1f}%")

    if model_type == "hmm" and "transition_matrix" in metadata:
        transition = np.array(metadata["transition_matrix"])
        current_state = int(latest["state"])
        print("\nNext-step transition tendency:")
        print(f"  Stay in state {current_state}: {transition[current_state, current_state] * 100:.1f}%")

    return result


def plot_market_prediction(df, window=365):
    if df is None or df.empty:
        print("\nSkipped market classification chart: no prediction rows available.")
        return

    plot_df = df.tail(window).copy()
    if plot_df.empty:
        print("\nSkipped market classification chart: no rows in the selected window.")
        return

    fig, ax = plt.subplots(figsize=(15, 5.0), facecolor="#EEF1F4")
    ax.set_facecolor("#F6F8FA")

    datetimes = plot_df["datetime"].to_numpy()
    closes = plot_df["close"].to_numpy()
    x_values = mdates.date2num(datetimes)

    ax.plot(
        datetimes,
        closes,
        color="#111827",
        linewidth=1.4,
        alpha=0.95,
        label="BTC Price",
        zorder=4,
    )
    ax.set_ylabel("Price (USD)")
    ax.set_title("BTC Spot Price With Market Regime Transitions")
    ax.grid(True, axis="y", alpha=0.18, linewidth=0.8)
    ax.grid(False, axis="x")
    ax.margins(y=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B8C0CC")
    ax.spines["bottom"].set_color("#B8C0CC")
    ax.tick_params(colors="#475467")

    y_min = float(plot_df["close"].min())
    y_max = float(plot_df["close"].max())
    y_padding = (y_max - y_min) * 0.16 if y_max > y_min else y_max * 0.08
    baseline = y_min - y_padding * 0.35

    transition_points = []
    full_span_segments = list(_iter_regime_segments(plot_df))
    first_segment = True
    for regime_code, start_time, end_time in full_span_segments:
        segment_df = plot_df[(plot_df["datetime"] >= start_time) & (plot_df["datetime"] <= end_time)]
        if len(segment_df) >= 2:
            ax.fill_between(
                segment_df["datetime"],
                segment_df["close"],
                baseline,
                color=REGIME_COLORS[int(regime_code)],
                alpha=0.15 if int(regime_code) != 0 else 0.12,
                linewidth=0,
                zorder=1,
            )
            ax.plot(
                segment_df["datetime"],
                segment_df["close"],
                color=REGIME_COLORS[int(regime_code)],
                linewidth=3.2,
                alpha=0.98,
                solid_capstyle="round",
                zorder=5,
            )
        if first_segment:
            first_segment = False
            continue
        transition_row = plot_df.loc[plot_df["datetime"] == start_time].iloc[0]
        transition_points.append(
            (start_time, transition_row["close"], int(regime_code), REGIME_PLOT_LABELS[int(regime_code)])
        )

    legend_handles = [
        Patch(facecolor=REGIME_COLORS[regime_code], label=REGIME_PLOT_LABELS[regime_code])
        for regime_code in REGIME_LABELS
    ]
    ax.legend(
        handles=[ax.lines[0], *legend_handles],
        loc="upper left",
        ncol=4,
        frameon=True,
        fancybox=True,
        facecolor="white",
        edgecolor="#D0D5DD",
        framealpha=0.96,
    )

    if transition_points:
        for transition_time, transition_price, regime_code, regime_name in transition_points:
            ax.axvline(
                transition_time,
                color=REGIME_COLORS[regime_code],
                linestyle=(0, (3, 5)),
                linewidth=1.1,
                alpha=0.22,
                zorder=2,
            )
            ax.scatter(
                [transition_time],
                [transition_price],
                s=64,
                color=REGIME_COLORS[regime_code],
                edgecolors="white",
                linewidths=1.2,
                zorder=6,
            )

    latest = plot_df.iloc[-1]
    regime_share = plot_df["prediction"].value_counts(normalize=True).sort_index()
    avg_probs = {
        "Consolidation": plot_df["prob_consolidation"].mean() * 100,
        "Bull": plot_df["prob_bull"].mean() * 100,
        "Bear": plot_df["prob_bear"].mean() * 100,
    }
    share_text = "   ".join(
        f"{REGIME_PLOT_LABELS[code]} {regime_share.get(code, 0) * 100:.0f}%"
        for code in REGIME_LABELS
    )
    prob_text = "   ".join(f"{name} {value:.0f}%" for name, value in avg_probs.items())
    info_text = (
        f"Latest: {REGIME_PLOT_LABELS[int(latest['prediction'])]}\n"
        f"Window share: {share_text}\n"
        f"Avg prob: {prob_text}"
    )
    ax.text(
        0.995,
        0.98,
        info_text,
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#101828",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": REGIME_COLORS[int(latest["prediction"])],
            "linewidth": 1.6,
            "alpha": 0.96,
        },
    )

    ax.scatter(
        [plot_df["datetime"].iloc[-1]],
        [plot_df["close"].iloc[-1]],
        s=90,
        color=REGIME_COLORS[int(latest["prediction"])],
        edgecolors="white",
        linewidths=1.5,
        zorder=6,
    )

    ax.set_ylim(baseline, y_max + y_padding * 0.45)

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.xticks(rotation=45)
    plt.tight_layout()
    os.makedirs("results", exist_ok=True)
    plt.savefig("results/market_classification.png", dpi=150)
    plt.close()
    print("\nSaved market classification chart to results/market_classification.png")


def analyze_recent_prediction(df, days=30):
    if days <= 0:
        print("\nSkipped recent regime summary: days must be positive.")
        return

    recent = df.tail(days).copy()
    if recent.empty:
        print("\nSkipped recent regime summary: no prediction rows available.")
        return

    print(f"\nRecent {days}-day regime summary:")
    print("-" * 40)
    for regime_code, label in REGIME_LABELS.items():
        count = (recent["prediction"] == regime_code).sum()
        share = count / len(recent) * 100
        bar = "#" * int(share / 2)
        print(f"{label:25s}: {bar} {share:.1f}%")

    print("\nAverage posterior probabilities:")
    print(f"  Consolidation: {recent['prob_consolidation'].mean() * 100:.1f}%")
    print(f"  Bull: {recent['prob_bull'].mean() * 100:.1f}%")
    print(f"  Bear: {recent['prob_bear'].mean() * 100:.1f}%")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BTC market regime inference")
    parser.add_argument("--symbol", type=str, default="BTC/USDT", help="Trading pair")
    parser.add_argument("--exchange", type=str, default="binance", help="Exchange name")
    parser.add_argument("--days", type=positive_int, default=365, help="Number of recent days to display")
    parser.add_argument("--timeframe", type=str, default="4h", help="Candle timeframe for --update mode")
    parser.add_argument("--update", action="store_true", help="Fetch the latest exchange data before inference")
    args = parser.parse_args()

    if args.update:
        result = get_current_status(args.symbol, args.exchange, args.timeframe)
    else:
        if not os.path.exists("models/model.pkl"):
            print("Model not found. Please run train.py first.")
            raise SystemExit(1)

        df = load_data()
        if df is None:
            print("Please run collect_data.py first, or use --update.")
            raise SystemExit(1)

        model, scaler, feature_cols, model_type, state_to_regime, metadata = load_model()
        result = predict(df, model, scaler, feature_cols, model_type, state_to_regime, metadata)

    if result is None:
        raise SystemExit(1)

    if result.empty:
        if not args.update:
            print(EMPTY_RESULT_MESSAGE)
        raise SystemExit(1)

    plot_market_prediction(result, window=args.days)
    analyze_recent_prediction(result, days=args.days)
