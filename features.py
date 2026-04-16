import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from ta.momentum import ROCIndicator, RSIIndicator, StochasticOscillator, WilliamsRIndicator
from ta.trend import ADXIndicator, EMAIndicator, IchimokuIndicator, MACD, SMAIndicator
from ta.volatility import AverageTrueRange, BollingerBands
from ta.volume import OnBalanceVolumeIndicator


def _safe_divide(numerator, denominator):
    """Avoid division by zero and inf values."""
    if isinstance(denominator, (int, float, np.integer, np.floating)):
        result = np.full_like(numerator, np.nan, dtype=float)
        mask = denominator != 0
        result[mask] = numerator[mask] / denominator if hasattr(numerator, "__getitem__") else numerator / denominator
        return result
    denominator = denominator.replace(0, np.nan)
    return numerator / denominator


def _infer_bars_per_day(df):
    """Infer candle frequency from datetime spacing for timeframe-aware features."""
    if "datetime" not in df.columns or len(df) < 2:
        return 1

    datetimes = pd.to_datetime(df["datetime"], errors="coerce")
    deltas = datetimes.diff().dropna()
    if deltas.empty:
        return 1

    median_seconds = deltas.dt.total_seconds().median()
    if pd.isna(median_seconds) or median_seconds <= 0:
        return 1

    bars_per_day = int(round(86400 / median_seconds))
    return max(1, bars_per_day)


def _days_to_bars(days, bars_per_day):
    return max(1, int(round(days * bars_per_day)))


def compute_features(df):
    """Compute normalized technical features for regime analysis."""
    df = df.copy()
    bars_per_day = _infer_bars_per_day(df)

    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]
    volume = df["volume"]

    return_1d_bars = _days_to_bars(1, bars_per_day)
    return_7d_bars = _days_to_bars(7, bars_per_day)
    return_30d_bars = _days_to_bars(30, bars_per_day)
    return_90d_bars = _days_to_bars(90, bars_per_day)

    df["returns_1d"] = close.pct_change(return_1d_bars)
    df["returns_7d"] = close.pct_change(return_7d_bars)
    df["returns_30d"] = close.pct_change(return_30d_bars)
    df["returns_90d"] = close.pct_change(return_90d_bars)

    bar_returns = close.pct_change(1)
    vol_7d_bars = _days_to_bars(7, bars_per_day)
    vol_30d_bars = _days_to_bars(30, bars_per_day)
    df["volatility_7d"] = bar_returns.rolling(window=vol_7d_bars).std()
    df["volatility_30d"] = bar_returns.rolling(window=vol_30d_bars).std()
    df["hist_vol_30d"] = df["volatility_30d"] * np.sqrt(365 * bars_per_day)
    df["vol_compression_7_30"] = _safe_divide(df["volatility_7d"], df["volatility_30d"])

    df["sma_20"] = SMAIndicator(close, window=20).sma_indicator()
    df["sma_50"] = SMAIndicator(close, window=50).sma_indicator()
    df["sma_200"] = SMAIndicator(close, window=200).sma_indicator()
    df["ema_12"] = EMAIndicator(close, window=12).ema_indicator()
    df["ema_26"] = EMAIndicator(close, window=26).ema_indicator()

    df["price_sma20_ratio"] = _safe_divide(close, df["sma_20"]) - 1
    df["price_sma50_ratio"] = _safe_divide(close, df["sma_50"]) - 1
    df["price_sma200_ratio"] = _safe_divide(close, df["sma_200"]) - 1
    df["sma_cross_diff"] = _safe_divide(df["sma_20"] - df["sma_50"], df["sma_50"])
    df["trend_strength_50_200"] = _safe_divide(df["sma_50"] - df["sma_200"], df["sma_200"])
    df["ema_gap_ratio"] = _safe_divide(df["ema_12"] - df["ema_26"], close)
    df["price_sma20_abs"] = df["price_sma20_ratio"].abs()
    df["ema_gap_abs"] = df["ema_gap_ratio"].abs()

    df["rsi_14"] = RSIIndicator(close, window=14).rsi() / 100
    df["rsi_28"] = RSIIndicator(close, window=28).rsi() / 100
    df["rsi_mid_distance"] = (df["rsi_14"] - 0.5).abs()

    macd = MACD(close)
    df["macd_ratio"] = _safe_divide(macd.macd(), close)
    df["macd_signal_ratio"] = _safe_divide(macd.macd_signal(), close)
    df["macd_hist_ratio"] = _safe_divide(macd.macd_diff(), close)
    df["macd_hist_abs"] = df["macd_hist_ratio"].abs()

    bb = BollingerBands(close, window=20, window_dev=2)
    df["bb_width"] = _safe_divide(
        bb.bollinger_hband() - bb.bollinger_lband(),
        bb.bollinger_mavg(),
    )
    df["bb_position"] = _safe_divide(
        close - bb.bollinger_lband(),
        bb.bollinger_hband() - bb.bollinger_lband(),
    )
    df["bb_mid_distance_abs"] = (df["bb_position"] - 0.5).abs()

    df["atr_ratio"] = _safe_divide(
        AverageTrueRange(high, low, close, window=14).average_true_range(),
        close,
    )
    df["atr_compression_14_30"] = _safe_divide(
        df["atr_ratio"],
        df["atr_ratio"].rolling(window=30).mean(),
    )

    stoch = StochasticOscillator(high, low, close)
    df["stoch_k"] = stoch.stoch() / 100
    df["stoch_d"] = stoch.stoch_signal() / 100
    df["williams_r"] = (WilliamsRIndicator(high, low, close, lbp=14).williams_r() + 100) / 100

    df["roc_12"] = ROCIndicator(close, window=12).roc() / 100
    df["roc_26"] = ROCIndicator(close, window=26).roc() / 100

    adx = ADXIndicator(high, low, close, window=14)
    df["adx"] = adx.adx() / 100
    df["adx_direction"] = (adx.adx_pos() - adx.adx_neg()) / 100

    obv = OnBalanceVolumeIndicator(close, volume).on_balance_volume()
    df["obv_pct_change"] = obv.pct_change().replace([np.inf, -np.inf], np.nan)

    df["volume_ma_20"] = volume.rolling(window=20).mean()
    df["volume_ratio"] = _safe_divide(volume, df["volume_ma_20"])

    df["high_low_spread"] = _safe_divide(high, low) - 1
    df["close_open_return"] = _safe_divide(close, open_) - 1
    df["price_momentum_14"] = close.pct_change(14)
    df["rolling_high_20"] = high.rolling(window=20).max()
    df["rolling_low_20"] = low.rolling(window=20).min()
    df["rolling_high_50"] = high.rolling(window=50).max()
    df["rolling_low_50"] = low.rolling(window=50).min()
    df["range_20_ratio"] = _safe_divide(df["rolling_high_20"] - df["rolling_low_20"], close)
    df["range_50_ratio"] = _safe_divide(df["rolling_high_50"] - df["rolling_low_50"], close)

    ichi = IchimokuIndicator(high, low, window1=9, window2=26, window3=52)
    ichi_tenkan = ichi.ichimoku_conversion_line()
    ichi_kijun = ichi.ichimoku_base_line()
    ichi_a = ichi.ichimoku_a()
    ichi_b = ichi.ichimoku_b()

    df["ichi_tenkan_ratio"] = _safe_divide(close - ichi_tenkan, close)
    df["ichi_kijun_ratio"] = _safe_divide(close - ichi_kijun, close)
    df["ichi_cloud_a_ratio"] = _safe_divide(close - ichi_a, close)
    df["ichi_cloud_b_ratio"] = _safe_divide(close - ichi_b, close)
    df["ichi_span_gap_ratio"] = _safe_divide(ichi_a - ichi_b, close)

    direction = np.sign(bar_returns)
    direction_change = (
        direction.ne(direction.shift(1)) & direction.ne(0) & direction.shift(1).ne(0)
    ).astype(float)
    df["direction_flip_rate_20"] = direction_change.rolling(window=20).mean()
    up_share_20 = (df["returns_1d"] > 0).rolling(window=20).mean()
    down_share_20 = (df["returns_1d"] < 0).rolling(window=20).mean()
    df["direction_balance_20"] = (up_share_20 - down_share_20).abs()

    return df


def get_feature_columns():
    """Feature columns shared by training and inference."""
    return [
        "price_sma20_ratio",
        "price_sma50_ratio",
        "price_sma200_ratio",
        "sma_cross_diff",
        "trend_strength_50_200",
        "ema_gap_ratio",
        "price_sma20_abs",
        "ema_gap_abs",
        "rsi_14",
        "rsi_28",
        "rsi_mid_distance",
        "macd_ratio",
        "macd_signal_ratio",
        "macd_hist_ratio",
        "macd_hist_abs",
        "bb_width",
        "bb_position",
        "bb_mid_distance_abs",
        "atr_ratio",
        "atr_compression_14_30",
        "stoch_k",
        "stoch_d",
        "williams_r",
        "roc_12",
        "roc_26",
        "adx",
        "adx_direction",
        "volume_ratio",
        "obv_pct_change",
        "returns_1d",
        "returns_7d",
        "returns_30d",
        "returns_90d",
        "volatility_7d",
        "volatility_30d",
        "hist_vol_30d",
        "vol_compression_7_30",
        "high_low_spread",
        "close_open_return",
        "price_momentum_14",
        "range_20_ratio",
        "range_50_ratio",
        "direction_flip_rate_20",
        "direction_balance_20",
        "ichi_tenkan_ratio",
        "ichi_kijun_ratio",
        "ichi_cloud_a_ratio",
        "ichi_cloud_b_ratio",
        "ichi_span_gap_ratio",
    ]


def get_hmm_feature_columns():
    """Compact HMM feature set with stronger consolidation signals.

    Filtered by correlation analysis to remove redundant features:
    - Short-term return (1d) removed; 7d captures signal better
    - Volatility features reduced to single volatility_7d
    - All MA ratio features reduced to price_sma20_ratio
    - Abs/signed pairs (macd_hist_abs, ema_gap_abs, etc.) reduced to one
    - Overlapping volatility measures (bb_width vs range_20_ratio) reduced
    - Direction features reduced to direction_flip_rate_20
    """
    return [
        "returns_7d",
        "returns_30d",
        "returns_90d",
        "volatility_7d",
        "price_sma20_ratio",
        "trend_strength_50_200",
        "ema_gap_ratio",
        "rsi_14",
        "rsi_mid_distance",
        "macd_hist_ratio",
        "bb_mid_distance_abs",
        "atr_ratio",
        "adx",
        "adx_direction",
        "volume_ratio",
        "range_20_ratio",
        "direction_flip_rate_20",
        "ichi_span_gap_ratio",
        "close_open_return",
    ]


HMM_FEATURE_CORRELATION_PAIRS = [
    ("returns_1d", "returns_7d"),
    ("volatility_30d", "volatility_7d"),
    ("hist_vol_30d", "volatility_7d"),
    ("vol_compression_7_30", "volatility_7d"),
    ("price_sma50_ratio", "price_sma20_ratio"),
    ("price_sma200_ratio", "price_sma20_ratio"),
    ("price_sma20_abs", "price_sma20_ratio"),
    ("sma_cross_diff", "price_sma20_ratio"),
    ("ema_gap_abs", "ema_gap_ratio"),
    ("macd_hist_abs", "macd_hist_ratio"),
    ("bb_width", "range_20_ratio"),
    ("atr_compression_14_30", "atr_ratio"),
    ("price_momentum_14", "returns_7d"),
    ("range_50_ratio", "range_20_ratio"),
    ("direction_balance_20", "direction_flip_rate_20"),
]


def build_feature_frame(df, feature_cols):
    """Keep only complete finite feature rows without future-value filling."""
    missing_cols = [c for c in feature_cols if c not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing columns in dataframe: {missing_cols}")
    if not feature_cols:
        raise ValueError("feature_cols cannot be empty")
    feature_df = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    valid_mask = feature_df.notna().all(axis=1)
    return df.loc[valid_mask].copy(), feature_df.loc[valid_mask].copy()


def fit_scaler(feature_df, scaler_type="standard"):
    """Fit the scaler on training features only."""
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = StandardScaler()

    X = scaler.fit_transform(feature_df)
    return X, scaler


def transform_features(feature_df, scaler):
    """Transform features with an existing scaler."""
    return scaler.transform(feature_df)
