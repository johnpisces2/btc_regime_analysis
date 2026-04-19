import argparse
import itertools
import os
from types import SimpleNamespace

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import backtest as bt


TRAIN_START = "2021-01-01"
TRAIN_END = "2024-09-08 20:00"
HOLDOUT_START = "2024-09-09 00:00"
FULL_END = "2026-04-19"

MIN_REGIME_BARS_GRID = [3, 6, 12, 18, 24]
REBALANCE_THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.30]
MAX_WEIGHT_STEP_GRID = [0.25, 0.50, 1.00]
TRANSITION_COOLDOWN_GRID = [0, 3, 6, 12]
WEIGHT_GRID = {
    "Bull": [0.75, 0.90, 1.00],
    "Consolidation": [0.25, 0.50, 0.75],
    "Bear": [0.00, 0.10, 0.20, 0.35],
}
COST_SCENARIOS = [
    ("base", 0.0005, 10.0),
    ("high", 0.0010, 25.0),
    ("very_high", 0.0015, 50.0),
]
DATE_SCENARIOS = [
    ("full_2021", "2021-01-01", FULL_END),
    ("post_cycle_2022", "2022-01-01", FULL_END),
    ("recent_bull_2023", "2023-01-01", FULL_END),
    ("holdout_2024", HOLDOUT_START, FULL_END),
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Research whether the regime-driven BTC strategy has beat-B&H edge."
    )
    parser.add_argument("--input", default="results/regime_predictions_4h.csv")
    parser.add_argument("--output-dir", default="results")
    parser.add_argument("--initial-cash", type=float, default=100000.0)
    parser.add_argument("--fee-rate", type=float, default=0.0005)
    parser.add_argument("--slippage-bps", type=float, default=10.0)
    parser.add_argument(
        "--full-grid",
        action="store_true",
        help="Run the full parameter product. This can be slow.",
    )
    parser.add_argument(
        "--top-weight-seeds",
        type=int,
        default=3,
        help="Number of train-selected weight seeds expanded into the management grid.",
    )
    parser.add_argument(
        "--top-holdout",
        type=int,
        default=5,
        help="Number of train-selected candidates manually checked on holdout.",
    )
    return parser.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def make_backtest_args(
    *,
    initial_cash,
    strategy_preset,
    bull_weight,
    consolidation_weight,
    bear_weight,
    min_regime_bars,
    rebalance_threshold,
    max_weight_step,
    transition_cooldown_bars,
    fee_rate,
    slippage_bps,
    start_date=None,
    end_date=None,
):
    return SimpleNamespace(
        input="",
        datetime_col="datetime",
        open_col="open",
        close_col="close",
        regime_col="prediction_name",
        initial_cash=float(initial_cash),
        fee_rate=float(fee_rate),
        slippage_bps=float(slippage_bps),
        rebalance_threshold=float(rebalance_threshold),
        max_weight_step=float(max_weight_step),
        transition_cooldown_bars=int(transition_cooldown_bars),
        min_regime_bars=int(min_regime_bars),
        confidence_threshold=None,
        strategy_preset=strategy_preset,
        bull_weight=float(bull_weight),
        consolidation_weight=float(consolidation_weight),
        bear_weight=float(bear_weight),
        start_date=start_date,
        end_date=end_date,
        summary_output="",
        trades_output="",
        equity_output="",
        rules_output="",
        plot_output="",
        dca_monthly_investment=None,
        dca_external_cash=None,
    )


def prepare_window(raw_df, start_date, end_date, initial_cash, fee_rate, slippage_bps):
    args = make_backtest_args(
        initial_cash=initial_cash,
        strategy_preset="default",
        bull_weight=bt.DEFAULT_REGIME_WEIGHTS["Bull"],
        consolidation_weight=bt.DEFAULT_REGIME_WEIGHTS["Consolidation"],
        bear_weight=bt.DEFAULT_REGIME_WEIGHTS["Bear"],
        min_regime_bars=6,
        rebalance_threshold=0.20,
        max_weight_step=0.25,
        transition_cooldown_bars=6,
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
        start_date=start_date,
        end_date=end_date,
    )
    frame = bt.prepare_frame(raw_df, args)
    bars_per_day = bt.infer_bars_per_day(frame, args.datetime_col)
    return frame, bars_per_day


def candidate_id(candidate):
    return (
        f"{candidate['strategy_preset']}"
        f"_b{candidate['bull_weight']:.2f}"
        f"_c{candidate['consolidation_weight']:.2f}"
        f"_r{candidate['bear_weight']:.2f}"
        f"_m{candidate['min_regime_bars']}"
        f"_th{candidate['rebalance_threshold']:.2f}"
        f"_step{candidate['max_weight_step']:.2f}"
        f"_cool{candidate['transition_cooldown_bars']}"
    )


def baseline_candidates():
    weights = bt.DEFAULT_REGIME_WEIGHTS
    base = {
        "bull_weight": weights["Bull"],
        "consolidation_weight": weights["Consolidation"],
        "bear_weight": weights["Bear"],
        "min_regime_bars": 6,
        "rebalance_threshold": 0.20,
        "max_weight_step": 0.25,
        "transition_cooldown_bars": 6,
    }
    return [
        {"strategy_preset": "default", "stage": "baseline", **base},
        {"strategy_preset": "regime_baseline", "stage": "baseline", **base},
        {"strategy_preset": "no_transition", "stage": "baseline", **base},
        {"strategy_preset": "transition_confirmed", "stage": "baseline", **base},
    ]


def iter_weight_candidates():
    for preset in ["custom_weights", "transition_confirmed"]:
        for bull, consolidation, bear in itertools.product(
            WEIGHT_GRID["Bull"],
            WEIGHT_GRID["Consolidation"],
            WEIGHT_GRID["Bear"],
        ):
            yield {
                "strategy_preset": preset,
                "stage": "weight",
                "bull_weight": bull,
                "consolidation_weight": consolidation,
                "bear_weight": bear,
                "min_regime_bars": 6,
                "rebalance_threshold": 0.20,
                "max_weight_step": 0.25,
                "transition_cooldown_bars": 6,
            }


def iter_management_candidates(seed_candidates):
    for seed in seed_candidates:
        for min_bars, threshold, max_step, cooldown in itertools.product(
            MIN_REGIME_BARS_GRID,
            REBALANCE_THRESHOLD_GRID,
            MAX_WEIGHT_STEP_GRID,
            TRANSITION_COOLDOWN_GRID,
        ):
            candidate = dict(seed)
            candidate.update(
                {
                    "stage": "management",
                    "min_regime_bars": min_bars,
                    "rebalance_threshold": threshold,
                    "max_weight_step": max_step,
                    "transition_cooldown_bars": cooldown,
                }
            )
            yield candidate


def iter_full_grid_candidates():
    for preset in ["custom_weights", "transition_confirmed"]:
        for bull, consolidation, bear, min_bars, threshold, max_step, cooldown in itertools.product(
            WEIGHT_GRID["Bull"],
            WEIGHT_GRID["Consolidation"],
            WEIGHT_GRID["Bear"],
            MIN_REGIME_BARS_GRID,
            REBALANCE_THRESHOLD_GRID,
            MAX_WEIGHT_STEP_GRID,
            TRANSITION_COOLDOWN_GRID,
        ):
            yield {
                "strategy_preset": preset,
                "stage": "full_grid",
                "bull_weight": bull,
                "consolidation_weight": consolidation,
                "bear_weight": bear,
                "min_regime_bars": min_bars,
                "rebalance_threshold": threshold,
                "max_weight_step": max_step,
                "transition_cooldown_bars": cooldown,
            }


def run_candidate(frame, bars_per_day, candidate, initial_cash, fee_rate, slippage_bps):
    args = make_backtest_args(
        initial_cash=initial_cash,
        strategy_preset=candidate["strategy_preset"],
        bull_weight=candidate["bull_weight"],
        consolidation_weight=candidate["consolidation_weight"],
        bear_weight=candidate["bear_weight"],
        min_regime_bars=candidate["min_regime_bars"],
        rebalance_threshold=candidate["rebalance_threshold"],
        max_weight_step=candidate["max_weight_step"],
        transition_cooldown_bars=candidate["transition_cooldown_bars"],
        fee_rate=fee_rate,
        slippage_bps=slippage_bps,
    )
    equity_df, trades_df = bt.run_backtest(frame, args)
    summary_df, equity_df = bt.compute_performance_metrics(
        equity_df=equity_df,
        trades_df=trades_df,
        initial_cash=initial_cash,
        bars_per_day=bars_per_day,
        args=args,
    )
    summary = dict(zip(summary_df["metric"], summary_df["value"]))
    years = float(summary["years"])
    trade_count = int(float(summary["trade_count"]))
    final_equity = float(summary["final_equity_regime"])
    buy_hold_final = float(summary["final_equity_buy_hold"])
    cagr = float(summary["cagr_regime"])
    buy_hold_cagr = float(summary["cagr_buy_hold"])
    max_dd = float(summary["max_drawdown_regime"])
    buy_hold_dd = float(summary["max_drawdown_buy_hold"])
    total_turnover = float(summary["total_turnover"])
    row = {
        "strategy_id": candidate_id(candidate),
        "stage": candidate.get("stage", ""),
        **candidate,
        "final_equity_regime": final_equity,
        "final_equity_buy_hold": buy_hold_final,
        "excess_final_equity": final_equity - buy_hold_final,
        "cagr_regime": cagr,
        "cagr_buy_hold": buy_hold_cagr,
        "excess_cagr": cagr - buy_hold_cagr,
        "total_return_regime": float(summary["total_return_regime"]),
        "total_return_buy_hold": float(summary["total_return_buy_hold"]),
        "max_drawdown_regime": max_dd,
        "max_drawdown_buy_hold": buy_hold_dd,
        "sharpe_regime": float(summary["sharpe_regime"]),
        "sharpe_buy_hold": float(summary["sharpe_buy_hold"]),
        "trade_count": trade_count,
        "annual_trade_count": trade_count / max(years, 1e-9),
        "transition_triggered_trade_count": int(
            float(summary["transition_triggered_trade_count"])
        ),
        "total_turnover": total_turnover,
        "turnover_to_initial_cash": total_turnover / initial_cash,
        "total_fees_paid": float(summary["total_fees_paid"]),
        "avg_spot_weight": float(summary["avg_spot_weight"]),
        "years": years,
        "beats_buy_hold": bool(final_equity > buy_hold_final or cagr > buy_hold_cagr),
        "drawdown_not_worse": bool(max_dd >= buy_hold_dd),
        "risk_overlay": bool(final_equity <= buy_hold_final and max_dd > buy_hold_dd),
    }
    return row, equity_df, trades_df


def select_candidates(train_df, limit):
    eligible = train_df.loc[
        (train_df["turnover_to_initial_cash"] <= 100.0)
        & (train_df["annual_trade_count"] <= 150.0)
    ].copy()
    if eligible.empty:
        eligible = train_df.copy()
    return eligible.sort_values(
        ["excess_cagr", "final_equity_regime", "max_drawdown_regime", "total_turnover"],
        ascending=[False, False, False, True],
    ).head(limit)


def run_train_search(train_frame, train_bars, args):
    rows = []
    print("Running baseline and weight-stage train candidates...", flush=True)
    stage_one_candidates = baseline_candidates() + list(iter_weight_candidates())
    for idx, candidate in enumerate(stage_one_candidates, start=1):
        row, _, _ = run_candidate(
            train_frame,
            train_bars,
            candidate,
            args.initial_cash,
            args.fee_rate,
            args.slippage_bps,
        )
        rows.append(row)
        if idx % 25 == 0 or idx == len(stage_one_candidates):
            print(f"  stage 1: {idx}/{len(stage_one_candidates)}", flush=True)

    stage_one_df = pd.DataFrame(rows)
    if args.full_grid:
        management_candidates = list(iter_full_grid_candidates())
    else:
        seed_rows = select_candidates(stage_one_df, args.top_weight_seeds)
        seed_candidates = [
            {
                "strategy_preset": row["strategy_preset"],
                "bull_weight": row["bull_weight"],
                "consolidation_weight": row["consolidation_weight"],
                "bear_weight": row["bear_weight"],
            }
            for _, row in seed_rows.iterrows()
        ]
        management_candidates = list(iter_management_candidates(seed_candidates))

    print(f"Running management-stage train candidates: {len(management_candidates)}", flush=True)
    for idx, candidate in enumerate(management_candidates, start=1):
        row, _, _ = run_candidate(
            train_frame,
            train_bars,
            candidate,
            args.initial_cash,
            args.fee_rate,
            args.slippage_bps,
        )
        rows.append(row)
        if idx % 250 == 0 or idx == len(management_candidates):
            print(f"  stage 2: {idx}/{len(management_candidates)}", flush=True)

    return pd.DataFrame(rows).drop_duplicates("strategy_id")


def run_holdout_for_top(train_results, holdout_frame, holdout_bars, args):
    top_train = select_candidates(train_results, args.top_holdout)
    rows = []
    equities = {}
    trades = {}
    for _, candidate in top_train.iterrows():
        candidate_dict = {
            "strategy_preset": candidate["strategy_preset"],
            "stage": "holdout_top_train",
            "bull_weight": candidate["bull_weight"],
            "consolidation_weight": candidate["consolidation_weight"],
            "bear_weight": candidate["bear_weight"],
            "min_regime_bars": candidate["min_regime_bars"],
            "rebalance_threshold": candidate["rebalance_threshold"],
            "max_weight_step": candidate["max_weight_step"],
            "transition_cooldown_bars": candidate["transition_cooldown_bars"],
        }
        row, equity_df, trades_df = run_candidate(
            holdout_frame,
            holdout_bars,
            candidate_dict,
            args.initial_cash,
            args.fee_rate,
            args.slippage_bps,
        )
        rows.append(row)
        equities[row["strategy_id"]] = equity_df
        trades[row["strategy_id"]] = trades_df
    return pd.DataFrame(rows), equities, trades


def run_scenario_table(raw_df, candidate, scenarios, args, output_dir):
    rows = []
    for name, start_date, end_date in scenarios:
        frame, bars = prepare_window(
            raw_df,
            start_date,
            end_date,
            args.initial_cash,
            args.fee_rate,
            args.slippage_bps,
        )
        row, _, _ = run_candidate(
            frame,
            bars,
            {**candidate, "stage": name},
            args.initial_cash,
            args.fee_rate,
            args.slippage_bps,
        )
        row["scenario"] = name
        row["start_date"] = start_date
        row["end_date"] = end_date
        rows.append(row)
    scenario_df = pd.DataFrame(rows)
    scenario_df.to_csv(
        os.path.join(output_dir, "strategy_research_date_robustness.csv"),
        index=False,
    )
    return scenario_df


def run_cost_table(holdout_frame, holdout_bars, candidate, args, output_dir):
    rows = []
    for name, fee_rate, slippage_bps in COST_SCENARIOS:
        row, _, _ = run_candidate(
            holdout_frame,
            holdout_bars,
            {**candidate, "stage": f"cost_{name}"},
            args.initial_cash,
            fee_rate,
            slippage_bps,
        )
        row["cost_scenario"] = name
        row["fee_rate"] = fee_rate
        row["slippage_bps"] = slippage_bps
        rows.append(row)
    cost_df = pd.DataFrame(rows)
    cost_df.to_csv(
        os.path.join(output_dir, "strategy_research_cost_stress.csv"),
        index=False,
    )
    return cost_df


def plot_top_holdout(equities, output_path):
    if not equities:
        return
    fig, ax = plt.subplots(figsize=(13, 6))
    first_equity = next(iter(equities.values()))
    ax.plot(
        first_equity["datetime"],
        first_equity["buy_hold_equity"],
        label="Buy & Hold",
        color="#8B0000",
        linewidth=1.5,
        alpha=0.85,
    )
    for idx, (strategy_id, equity_df) in enumerate(equities.items(), start=1):
        ax.plot(
            equity_df["datetime"],
            equity_df["equity"],
            label=f"Top {idx}: {strategy_id[:42]}",
            linewidth=1.2,
            alpha=0.9,
        )
    ax.set_title("Top Train Candidates on Holdout")
    ax.set_ylabel("Equity")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def _fmt_pct(value):
    return f"{value * 100:.2f}%"


def markdown_table(df, columns, max_rows=10):
    if df.empty:
        return "_No rows._"
    view = df.loc[:, columns].head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.4f}")
    rows = []
    headers = [str(col) for col in view.columns]
    rows.append("| " + " | ".join(headers) + " |")
    rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for _, row in view.iterrows():
        values = [str(row[col]) for col in view.columns]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def write_report(
    train_results,
    holdout_results,
    cost_df,
    scenario_df,
    output_path,
):
    best_holdout = holdout_results.sort_values(
        ["excess_cagr", "final_equity_regime"],
        ascending=[False, False],
    ).iloc[0]
    edge_found = bool((
        (holdout_results["final_equity_regime"] > holdout_results["final_equity_buy_hold"])
        | (holdout_results["cagr_regime"] > holdout_results["cagr_buy_hold"])
    ).any())
    best_is_regime_only = best_holdout["strategy_preset"] in {
        "custom_weights",
        "regime_baseline",
        "no_transition",
    }

    recommendation = "未找到 beat-B&H edge。"
    if edge_found and best_is_regime_only:
        recommendation = "Holdout 最佳組合打贏 B&H，且 regime-only 打贏 transition rules；建議改用簡化的 regime-only 權重策略。"
    elif edge_found:
        recommendation = "Holdout 最佳組合打贏 B&H；可保留 confirmed-transition 架構，但只採用 holdout 驗證過的參數。"
    elif (holdout_results["risk_overlay"]).any():
        recommendation = "未找到 beat-B&H edge，但部分組合降低 drawdown；只能定位為 risk overlay，不應宣稱能打贏 B&H。"

    lines = [
        "# Regime Strategy Research Report",
        "",
        "## Verdict",
        "",
        f"- Beat-B&H edge found: **{'YES' if edge_found else 'NO'}**",
        f"- Recommendation: **{recommendation}**",
        f"- Best holdout strategy: `{best_holdout['strategy_id']}`",
        f"- Best holdout final equity: `${best_holdout['final_equity_regime']:,.2f}` vs B&H `${best_holdout['final_equity_buy_hold']:,.2f}`",
        f"- Best holdout CAGR: {_fmt_pct(best_holdout['cagr_regime'])} vs B&H {_fmt_pct(best_holdout['cagr_buy_hold'])}",
        f"- Best holdout max drawdown: {_fmt_pct(best_holdout['max_drawdown_regime'])} vs B&H {_fmt_pct(best_holdout['max_drawdown_buy_hold'])}",
        "",
    ]
    if not edge_found:
        lines.extend(
            [
                "> 未找到 beat-B&H edge：train 選出的 top candidates 在 holdout final equity/CAGR 未能可靠勝過 buy-and-hold。",
                "",
            ]
        )

    train_cols = [
        "strategy_id",
        "stage",
        "excess_cagr",
        "final_equity_regime",
        "final_equity_buy_hold",
        "max_drawdown_regime",
        "trade_count",
        "turnover_to_initial_cash",
    ]
    holdout_cols = [
        "strategy_id",
        "strategy_preset",
        "excess_cagr",
        "final_equity_regime",
        "final_equity_buy_hold",
        "max_drawdown_regime",
        "trade_count",
        "turnover_to_initial_cash",
        "beats_buy_hold",
        "risk_overlay",
    ]
    lines.extend(
        [
            "## Top Train Candidates",
            "",
            markdown_table(select_candidates(train_results, 10), train_cols),
            "",
            "## Holdout Results for Top Train Candidates",
            "",
            markdown_table(
                holdout_results.sort_values(["excess_cagr"], ascending=False),
                holdout_cols,
            ),
            "",
            "## Cost Stress on Best Holdout Candidate",
            "",
            markdown_table(
                cost_df,
                [
                    "cost_scenario",
                    "fee_rate",
                    "slippage_bps",
                    "excess_cagr",
                    "final_equity_regime",
                    "final_equity_buy_hold",
                    "max_drawdown_regime",
                    "trade_count",
                    "total_fees_paid",
                ],
            ),
            "",
            "## Date Robustness on Best Holdout Candidate",
            "",
            markdown_table(
                scenario_df,
                [
                    "scenario",
                    "start_date",
                    "excess_cagr",
                    "final_equity_regime",
                    "final_equity_buy_hold",
                    "max_drawdown_regime",
                    "trade_count",
                    "turnover_to_initial_cash",
                ],
            ),
            "",
            "## Notes",
            "",
            "- Train selection filters out candidates with turnover over 100x initial cash or annual trade count over 150.",
            "- Full-window results are diagnostic only and are not used for parameter selection.",
            "- Strategy remains long-only spot/cash; no shorting, leverage, stops, or take-profit rules are included.",
            "",
        ]
    )
    with open(output_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines))


def candidate_from_row(row):
    return {
        "strategy_preset": row["strategy_preset"],
        "bull_weight": float(row["bull_weight"]),
        "consolidation_weight": float(row["consolidation_weight"]),
        "bear_weight": float(row["bear_weight"]),
        "min_regime_bars": int(row["min_regime_bars"]),
        "rebalance_threshold": float(row["rebalance_threshold"]),
        "max_weight_step": float(row["max_weight_step"]),
        "transition_cooldown_bars": int(row["transition_cooldown_bars"]),
    }


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    raw_df = bt.load_prediction_frame(args.input)

    train_frame, train_bars = prepare_window(
        raw_df,
        TRAIN_START,
        TRAIN_END,
        args.initial_cash,
        args.fee_rate,
        args.slippage_bps,
    )
    holdout_frame, holdout_bars = prepare_window(
        raw_df,
        HOLDOUT_START,
        FULL_END,
        args.initial_cash,
        args.fee_rate,
        args.slippage_bps,
    )

    train_results = run_train_search(train_frame, train_bars, args)
    train_results.to_csv(
        os.path.join(args.output_dir, "strategy_research_train.csv"),
        index=False,
    )

    holdout_results, equities, _ = run_holdout_for_top(
        train_results,
        holdout_frame,
        holdout_bars,
        args,
    )
    holdout_results.to_csv(
        os.path.join(args.output_dir, "strategy_research_holdout.csv"),
        index=False,
    )

    best_holdout = holdout_results.sort_values(
        ["excess_cagr", "final_equity_regime"],
        ascending=[False, False],
    ).iloc[0]
    best_candidate = candidate_from_row(best_holdout)
    cost_df = run_cost_table(
        holdout_frame,
        holdout_bars,
        best_candidate,
        args,
        args.output_dir,
    )
    scenario_df = run_scenario_table(
        raw_df,
        best_candidate,
        DATE_SCENARIOS,
        args,
        args.output_dir,
    )

    plot_top_holdout(
        equities,
        os.path.join(args.output_dir, "strategy_research_top_equity.png"),
    )
    write_report(
        train_results,
        holdout_results,
        cost_df,
        scenario_df,
        os.path.join(args.output_dir, "strategy_research_report.md"),
    )

    print("\nResearch complete.")
    print(f"Train rows: {len(train_results)}")
    print(f"Holdout rows: {len(holdout_results)}")
    print(f"Best holdout strategy: {best_holdout['strategy_id']}")
    print(
        "Best holdout final equity: "
        f"${best_holdout['final_equity_regime']:,.2f} vs "
        f"B&H ${best_holdout['final_equity_buy_hold']:,.2f}"
    )
    print(
        "Saved report to "
        f"{os.path.join(args.output_dir, 'strategy_research_report.md')}"
    )


if __name__ == "__main__":
    main()
