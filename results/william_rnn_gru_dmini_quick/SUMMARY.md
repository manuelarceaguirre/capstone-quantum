# William Sequence Models

- Models: rnn, gru
- Track: D-mini
- Sequence length: 24
- Fixed train window: 24 months
- Tested horizons: 1
- Folds available in substrate: 39 expanding-window folds with date ranges

## Best Rows

| target | horizon | model | rmse | mae | mase | n_predictions | mean_best_epoch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| INDPRO | 1 | gru | 0.006905 | 0.005443 | 0.912718 | 300 | 5.000000 |

## Notes

- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.
- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.
- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.
