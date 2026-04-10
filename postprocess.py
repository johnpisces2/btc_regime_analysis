import numpy as np


def _find_runs(labels):
    runs = []
    if len(labels) == 0:
        return runs

    start = 0
    current = labels[0]
    for idx in range(1, len(labels)):
        if labels[idx] != current:
            runs.append((start, idx, current))
            start = idx
            current = labels[idx]
    runs.append((start, len(labels), current))
    return runs


def smooth_regime_sequence(labels, min_run_length=6, max_passes=10):
    """Smooth short-lived regime flips."""
    arr = np.asarray(labels).astype(int).copy()
    if len(arr) == 0 or min_run_length <= 1:
        return arr

    for _ in range(max_passes):
        runs = _find_runs(arr)
        changed = False

        for idx, (start, end, label) in enumerate(runs):
            run_length = end - start
            if run_length >= min_run_length:
                continue

            prev_run = runs[idx - 1] if idx > 0 else None
            next_run = runs[idx + 1] if idx < len(runs) - 1 else None

            if prev_run and next_run and prev_run[2] == next_run[2]:
                arr[start:end] = prev_run[2]
                changed = True
                continue

            if prev_run and next_run:
                prev_length = prev_run[1] - prev_run[0]
                next_length = next_run[1] - next_run[0]
                arr[start:end] = prev_run[2] if prev_length >= next_length else next_run[2]
                changed = True
                continue

            if prev_run:
                arr[start:end] = prev_run[2]
                changed = True
                continue

            if next_run:
                arr[start:end] = next_run[2]
                changed = True

        if not changed:
            break

    return arr


def build_hmm_consolidation_rules(train_df):
    """Build feature-based rules that identify sideways / neutral conditions."""
    abs_return_30d = train_df["returns_30d"].abs()
    bear_subset = train_df.loc[train_df["regime"] == 2] if "regime" in train_df.columns else train_df.iloc[0:0]
    bull_subset = train_df.loc[train_df["regime"] == 1] if "regime" in train_df.columns else train_df.iloc[0:0]

    def _bear_quantile(column, subset_q, fallback_q, floor):
        subset = bear_subset[column].dropna()
        series = train_df[column].dropna()
        if len(subset) >= 100:
            value = float(subset.quantile(subset_q))
        elif len(series) > 0:
            value = float(series.quantile(fallback_q))
        else:
            value = floor
        return max(value, floor)

    def _bull_quantile(column, subset_q, fallback_q, floor):
        subset = bull_subset[column].dropna()
        series = train_df[column].dropna()
        if len(subset) >= 100:
            value = float(subset.quantile(subset_q))
        elif len(series) > 0:
            value = float(series.quantile(fallback_q))
        else:
            value = floor
        return min(value, floor)

    return {
        "adx_max": float(train_df["adx"].quantile(0.60)),
        "range_20_max": float(train_df["range_20_ratio"].quantile(0.60)),
        "bb_mid_distance_abs_max": float(train_df["bb_mid_distance_abs"].quantile(0.60)),
        "rsi_mid_distance_max": float(train_df["rsi_mid_distance"].quantile(0.65)),
        "abs_return_30d_max": float(abs_return_30d.quantile(0.60)),
        "flip_rate_min": float(train_df["direction_flip_rate_20"].quantile(0.45)),
        "neutral_min_conditions": 4,
        "entropy_threshold": 0.85,
        "consolidation_boost": 1.2,
        "consolidation_temperature": 0.30,
        "mask_min_run": 4,
        "bearish_return_7d_max": _bear_quantile("returns_7d", 0.70, 0.25, -0.015),
        "bearish_return_30d_max": _bear_quantile("returns_30d", 0.70, 0.25, -0.040),
        "bearish_price_sma20_ratio_max": _bear_quantile("price_sma20_ratio", 0.70, 0.25, -0.010),
        "bearish_ema_gap_ratio_max": _bear_quantile("ema_gap_ratio", 0.70, 0.25, -0.002),
        "bearish_macd_hist_ratio_max": _bear_quantile("macd_hist_ratio", 0.70, 0.25, -0.0008),
        "bearish_adx_direction_max": _bear_quantile("adx_direction", 0.70, 0.25, -0.010),
        "bearish_min_conditions": 4,
        "bear_priority_min_run": 3,
        "bear_priority_boost": 1.0,
        "directional_temperature": 0.20,
        "bullish_return_7d_min": _bull_quantile("returns_7d", 0.30, 0.75, 0.015),
        "bullish_return_30d_min": _bull_quantile("returns_30d", 0.30, 0.75, 0.040),
        "bullish_price_sma20_ratio_min": _bull_quantile("price_sma20_ratio", 0.30, 0.75, 0.010),
        "bullish_ema_gap_ratio_min": _bull_quantile("ema_gap_ratio", 0.30, 0.75, 0.002),
        "bullish_macd_hist_ratio_min": _bull_quantile("macd_hist_ratio", 0.30, 0.75, 0.0008),
        "bullish_adx_direction_min": _bull_quantile("adx_direction", 0.30, 0.75, 0.010),
        "bullish_min_conditions": 4,
        "bull_priority_min_run": 3,
        "bull_priority_boost": 1.0,
    }


def _compute_entropy(p):
    p = np.clip(p, 1e-10, 1.0)
    p = np.where(np.isnan(p), 1e-10, p)
    return -np.sum(p * np.log(p), axis=1)


def _softmax_adjustment(p, target_idx, boost_factor, temperature=0.25):
    logits = np.log(p + 1e-10)
    logits[:, target_idx] += boost_factor
    exp_logits = np.exp(logits / temperature)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


def apply_hmm_consolidation_override(df, regime_confidence, rules):
    """
    Intuition-based HMM postprocess:
    1. Use entropy to detect when model is uncertain
    2. Use neutral features to confirm consolidation signals
    3. Use directional features to override bear/bull when clear

    Entropy logic:
    - max entropy (1.1) = uniform distribution (33/33/33) -> high uncertainty
    - min entropy (0)   = one-hot distribution (100/0/0)  -> certain prediction
    """
    if not rules:
        return regime_confidence, np.zeros(len(df), dtype=bool)

    entropy = _compute_entropy(regime_confidence)
    max_prob = regime_confidence.max(axis=1)

    condition_matrix = np.column_stack(
        [
            df["adx"].to_numpy() <= rules["adx_max"],
            df["range_20_ratio"].to_numpy() <= rules["range_20_max"],
            df["bb_mid_distance_abs"].to_numpy() <= rules["bb_mid_distance_abs_max"],
            df["rsi_mid_distance"].to_numpy() <= rules["rsi_mid_distance_max"],
            np.abs(df["returns_30d"].to_numpy()) <= rules["abs_return_30d_max"],
            df["direction_flip_rate_20"].to_numpy() >= rules["flip_rate_min"],
        ]
    )
    neutral_score = condition_matrix.sum(axis=1)
    bear_condition_matrix = np.column_stack(
        [
            df["returns_7d"].to_numpy() <= rules["bearish_return_7d_max"],
            df["returns_30d"].to_numpy() <= rules["bearish_return_30d_max"],
            df["price_sma20_ratio"].to_numpy() <= rules["bearish_price_sma20_ratio_max"],
            df["ema_gap_ratio"].to_numpy() <= rules["bearish_ema_gap_ratio_max"],
            df["macd_hist_ratio"].to_numpy() <= rules["bearish_macd_hist_ratio_max"],
            df["adx_direction"].to_numpy() <= rules["bearish_adx_direction_max"],
        ]
    )
    bear_priority_mask = bear_condition_matrix.sum(axis=1) >= rules["bearish_min_conditions"]
    if rules.get("bear_priority_min_run", 1) > 1:
        bear_priority_mask = smooth_regime_sequence(
            bear_priority_mask.astype(int),
            min_run_length=int(rules["bear_priority_min_run"]),
        ).astype(bool)

    bull_condition_matrix = np.column_stack(
        [
            df["returns_7d"].to_numpy() >= rules["bullish_return_7d_min"],
            df["returns_30d"].to_numpy() >= rules["bullish_return_30d_min"],
            df["price_sma20_ratio"].to_numpy() >= rules["bullish_price_sma20_ratio_min"],
            df["ema_gap_ratio"].to_numpy() >= rules["bullish_ema_gap_ratio_min"],
            df["macd_hist_ratio"].to_numpy() >= rules["bullish_macd_hist_ratio_min"],
            df["adx_direction"].to_numpy() >= rules["bullish_adx_direction_min"],
        ]
    )
    bull_priority_mask = bull_condition_matrix.sum(axis=1) >= rules["bullish_min_conditions"]
    if rules.get("bull_priority_min_run", 1) > 1:
        bull_priority_mask = smooth_regime_sequence(
            bull_priority_mask.astype(int),
            min_run_length=int(rules["bull_priority_min_run"]),
        ).astype(bool)

    entropy_threshold = rules.get("entropy_threshold", 0.85)
    uncertain_mask = entropy >= entropy_threshold

    override_mask = (neutral_score >= rules["neutral_min_conditions"]) & uncertain_mask
    override_mask &= ~(bear_priority_mask | bull_priority_mask)

    if rules.get("mask_min_run", 1) > 1:
        override_mask = smooth_regime_sequence(
            override_mask.astype(int),
            min_run_length=int(rules["mask_min_run"]),
        ).astype(bool)

    adjusted = regime_confidence.copy()

    if override_mask.any():
        override_rows = adjusted[override_mask]
        logits = np.log(override_rows + 1e-10)
        boost = rules["consolidation_boost"]
        temp = rules.get("consolidation_temperature", 0.30)
        logits[:, 0] += boost
        exp_logits = np.exp(logits / temp)
        adjusted[override_mask] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    if bear_priority_mask.any():
        mask = bear_priority_mask & ((adjusted[:, 2] >= adjusted[:, 0]) | (adjusted[:, 2] >= 0.25))
        if mask.any():
            bear_rows = adjusted[mask]
            logits = np.log(bear_rows + 1e-10)
            boost = rules["bear_priority_boost"]
            temp = rules.get("directional_temperature", 0.20)
            logits[:, 2] += boost
            exp_logits = np.exp(logits / temp)
            adjusted[mask] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    if bull_priority_mask.any():
        mask = bull_priority_mask & ((adjusted[:, 1] >= adjusted[:, 0]) | (adjusted[:, 1] >= 0.25))
        if mask.any():
            bull_rows = adjusted[mask]
            logits = np.log(bull_rows + 1e-10)
            boost = rules["bull_priority_boost"]
            temp = rules.get("directional_temperature", 0.20)
            logits[:, 1] += boost
            exp_logits = np.exp(logits / temp)
            adjusted[mask] = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    return adjusted, override_mask, bear_priority_mask, bull_priority_mask
