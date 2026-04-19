# Regime Strategy Research Report

## Verdict

- Beat-B&H edge found: **NO**
- Recommendation: **未找到 beat-B&H edge，但部分組合降低 drawdown；只能定位為 risk overlay，不應宣稱能打贏 B&H。**
- Best holdout strategy: `custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool0`
- Best holdout final equity: `$122,086.55` vs B&H `$139,570.34`
- Best holdout CAGR: 13.24% vs B&H 23.09%
- Best holdout max drawdown: -18.40% vs B&H -49.84%

> 未找到 beat-B&H edge：train 選出的 top candidates 在 holdout final equity/CAGR 未能可靠勝過 buy-and-hold。

## Top Train Candidates

| strategy_id | stage | excess_cagr | final_equity_regime | final_equity_buy_hold | max_drawdown_regime | trade_count | turnover_to_initial_cash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| transition_confirmed_b0.75_c0.50_r0.35_m3_th0.30_step0.25_cool12 | management | 0.0745 | 232756.7980 | 185782.1158 | -0.5302 | 148 | 60.7592 |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool0 | management | 0.0627 | 224783.2615 | 185782.1158 | -0.3862 | 64 | 28.4343 |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool3 | management | 0.0627 | 224783.2615 | 185782.1158 | -0.3862 | 64 | 28.4343 |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool6 | management | 0.0627 | 224783.2615 | 185782.1158 | -0.3862 | 64 | 28.4343 |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool12 | management | 0.0627 | 224783.2615 | 185782.1158 | -0.3862 | 64 | 28.4343 |
| transition_confirmed_b0.75_c0.50_r0.35_m6_th0.20_step0.25_cool0 | management | 0.0600 | 222996.7762 | 185782.1158 | -0.4833 | 254 | 97.8379 |
| transition_confirmed_b0.75_c0.50_r0.35_m6_th0.20_step0.25_cool3 | management | 0.0571 | 221090.3865 | 185782.1158 | -0.4851 | 254 | 97.4488 |
| custom_weights_b0.75_c0.25_r0.35_m18_th0.20_step0.50_cool0 | management | 0.0534 | 218633.7347 | 185782.1158 | -0.3961 | 65 | 49.2978 |
| custom_weights_b0.75_c0.25_r0.35_m18_th0.20_step0.50_cool3 | management | 0.0534 | 218633.7347 | 185782.1158 | -0.3961 | 65 | 49.2978 |
| custom_weights_b0.75_c0.25_r0.35_m18_th0.20_step0.50_cool6 | management | 0.0534 | 218633.7347 | 185782.1158 | -0.3961 | 65 | 49.2978 |

## Holdout Results for Top Train Candidates

| strategy_id | strategy_preset | excess_cagr | final_equity_regime | final_equity_buy_hold | max_drawdown_regime | trade_count | turnover_to_initial_cash | beats_buy_hold | risk_overlay |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool0 | custom_weights | -0.0985 | 122086.5464 | 139570.3407 | -0.1840 | 31 | 9.5966 | False | True |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool3 | custom_weights | -0.0985 | 122086.5464 | 139570.3407 | -0.1840 | 31 | 9.5966 | False | True |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool6 | custom_weights | -0.0985 | 122086.5464 | 139570.3407 | -0.1840 | 31 | 9.5966 | False | True |
| custom_weights_b0.75_c0.25_r0.35_m24_th0.20_step0.25_cool12 | custom_weights | -0.0985 | 122086.5464 | 139570.3407 | -0.1840 | 31 | 9.5966 | False | True |
| transition_confirmed_b0.75_c0.50_r0.35_m3_th0.30_step0.25_cool12 | transition_confirmed | -0.1058 | 120826.0363 | 139570.3407 | -0.2772 | 48 | 15.6689 | False | True |

## Cost Stress on Best Holdout Candidate

| cost_scenario | fee_rate | slippage_bps | excess_cagr | final_equity_regime | final_equity_buy_hold | max_drawdown_regime | trade_count | total_fees_paid |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base | 0.0005 | 10.0000 | -0.0985 | 122086.5464 | 139570.3407 | -0.1840 | 31 | 479.8280 |
| high | 0.0010 | 25.0000 | -0.1092 | 120245.9930 | 139570.3407 | -0.1851 | 31 | 952.4574 |
| very_high | 0.0015 | 50.0000 | -0.1250 | 117531.7108 | 139570.3407 | -0.1873 | 31 | 1412.6322 |

## Date Robustness on Best Holdout Candidate

| scenario | start_date | excess_cagr | final_equity_regime | final_equity_buy_hold | max_drawdown_regime | trade_count | turnover_to_initial_cash |
| --- | --- | --- | --- | --- | --- | --- | --- |
| full_2021 | 2021-01-01 | 0.0229 | 290097.0883 | 262436.1414 | -0.3862 | 93 | 50.0639 |
| post_cycle_2022 | 2022-01-01 | -0.0278 | 148964.3917 | 165845.6211 | -0.2993 | 74 | 22.2547 |
| recent_bull_2023 | 2023-01-01 | -0.3408 | 206295.1734 | 457680.8041 | -0.2341 | 66 | 28.2151 |
| holdout_2024 | 2024-09-09 00:00 | -0.0985 | 122086.5464 | 139570.3407 | -0.1840 | 31 | 9.5966 |

## Notes

- Train selection filters out candidates with turnover over 100x initial cash or annual trade count over 150.
- Full-window results are diagnostic only and are not used for parameter selection.
- Strategy remains long-only spot/cash; no shorting, leverage, stops, or take-profit rules are included.
