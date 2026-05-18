# William Sequence Models

- Models: rnn, gru
- Track: B
- Sequence length: 24
- Fixed train window: 24 months
- Tested horizons: 1, 3, 6, 12
- Folds available in substrate: 39 expanding-window folds with date ranges

## Best Rows

| target | horizon | model | rmse | mae | mase | n_predictions | mean_best_epoch |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| CPIAUCSL | 1 | rnn | 0.002872 | 0.001990 | 1.152910 | 2340 | 15.179487 |
| CPIAUCSL | 3 | gru | 0.007130 | 0.004825 | 2.185147 | 2340 | 15.256410 |
| CPIAUCSL | 6 | gru | 0.011428 | 0.008095 | 3.551565 | 2340 | 14.641026 |
| CPIAUCSL | 12 | rnn | 0.018621 | 0.013652 | 5.754534 | 2339 | 13.769231 |
| INDPRO | 1 | rnn | 0.010444 | 0.005748 | 1.067959 | 2340 | 16.128205 |
| INDPRO | 3 | rnn | 0.021054 | 0.012888 | 2.407835 | 2340 | 15.153846 |
| INDPRO | 6 | gru | 0.033079 | 0.023051 | 3.913385 | 2340 | 16.179487 |
| INDPRO | 12 | gru | 0.054001 | 0.039088 | 5.771831 | 2339 | 17.025641 |
| PAYEMS | 1 | rnn | 0.007368 | 0.002024 | 2.388242 | 2340 | 16.102564 |
| PAYEMS | 3 | rnn | 0.012845 | 0.005502 | 6.320277 | 2340 | 15.461538 |
| PAYEMS | 6 | gru | 0.018458 | 0.010718 | 9.803875 | 2340 | 21.512821 |
| PAYEMS | 12 | gru | 0.029853 | 0.020596 | 15.198590 | 2339 | 24.692308 |
| S&P 500 | 1 | rnn | 0.038689 | 0.027981 | 0.948989 | 2340 | 10.923077 |
| S&P 500 | 3 | gru | 0.081012 | 0.059438 | 1.826745 | 2340 | 10.871795 |
| S&P 500 | 6 | gru | 0.132389 | 0.100089 | 2.993925 | 2340 | 14.692308 |
| S&P 500 | 12 | rnn | 0.199421 | 0.152069 | 4.352434 | 2339 | 18.205128 |

## Notes

- Training windows are intentionally fixed-length even though the shared substrate ships with expanding folds.
- The shared test periods stay unchanged, so comparison with other workstreams remains date-aligned.
- Track D-mini is re-fit fold by fold using the same anti-leakage rule from the README.
