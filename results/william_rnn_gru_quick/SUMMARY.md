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
| CPIAUCSL | 1 | gru | 0.003436 | 0.002421 | 1.245347 | 300 | 6.000000 |
| INDPRO | 1 | gru | 0.005914 | 0.004605 | 0.812262 | 300 | 3.000000 |
| PAYEMS | 1 | gru | 0.002626 | 0.001987 | 1.367355 | 300 | 4.000000 |
| S&P 500 | 1 | rnn | 0.038378 | 0.028265 | 0.861622 | 300 | 4.000000 |

## Notes

- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.
- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.
- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.
