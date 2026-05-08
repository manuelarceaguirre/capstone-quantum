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
| CPIAUCSL | 1 | gru | 0.003004 | 0.002183 | 2340 | 13.153846 |
| INDPRO | 1 | rnn | 0.010032 | 0.005375 | 2340 | 12.102564 |
| PAYEMS | 1 | rnn | 0.007219 | 0.001763 | 2340 | 11.846154 |
| S&P 500 | 1 | rnn | 0.037534 | 0.026956 | 2340 | 12.948718 |

## Notes

- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.
- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.
- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.
