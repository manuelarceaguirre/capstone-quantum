# William Sequence Models

- Models: rnn, gru
- Track: B
- Sequence length: 24
- Fixed train window: 24 months
- Tested horizons: 1
- Folds available in substrate: 39 expanding-window folds with date ranges

## Best Rows

| target | horizon | model | rmse | mae | mase | n_predictions | mean_best_epoch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CPIAUCSL | 1 | gru | 0.002917 | 0.002054 | 1.185069 | 2340 | 13.102564 |
| INDPRO | 1 | rnn | 0.010444 | 0.005748 | 1.067959 | 2340 | 16.128205 |
| PAYEMS | 1 | rnn | 0.007357 | 0.002013 | 2.373001 | 2340 | 16.410256 |
| S&P 500 | 1 | rnn | 0.038924 | 0.028287 | 0.958801 | 2340 | 11.512821 |

## Notes

- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.
- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.
- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.
