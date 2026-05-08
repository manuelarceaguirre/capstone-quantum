# William Sequence Models

- Models: rnn, gru
- Track: B
- Sequence length: 24
- Fixed train window: 120 months
- Tested horizons: 1
- Folds available in substrate: 39 expanding-window folds with date ranges

## Best Rows

| target | horizon | model | rmse | mae | n_predictions | mean_best_epoch |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| CPIAUCSL | 1 | gru | 0.004172 | 0.003626 | 300 | 5.000000 |
| INDPRO | 1 | rnn | 0.005202 | 0.004017 | 300 | 5.000000 |
| PAYEMS | 1 | rnn | 0.001579 | 0.001079 | 300 | 5.600000 |
| S&P 500 | 1 | rnn | 0.037660 | 0.028052 | 300 | 6.000000 |

## Notes

- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.
- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.
- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.
