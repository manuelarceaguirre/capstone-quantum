# William RNN/GRU Function Wrapper Report

## What I Built

I reviewed Manuel's `src/model_functions/manuel_konduri_functions.py` and matched his evaluator-facing pattern:

- each model is exposed as a small function;
- the function receives `train_df`, `test_df`, and the horizon-specific `y_train`;
- it trains internally and returns `(predictions, test_dates)`;
- the outer evaluation notebook remains responsible for metrics and test labels.

I then created `src/model_functions/william_rnn_gru_functions.py` with two callable wrappers:

- `rnn_model_function(train_df, test_df, y_train)`
- `gru_model_function(train_df, test_df, y_train)`

## Model Details

The wrappers preserve the main settings from `scripts/run_rnn_gru_benchmark.py`:

- 24-month input sequence
- hidden size 24
- 1 recurrent layer
- no dropout by default
- Adam optimizer
- MSE loss
- learning rate `1e-3`
- weight decay `1e-4`
- max 30 epochs
- early stopping patience 6
- gradient clipping at norm 1.0
- last 20% of training sequences used as validation

## Why This Matters

This turns William's RNN/GRU work from a standalone benchmark script into reusable model functions that can plug into the team's shared evaluation loop. It keeps the recurrent-model logic consistent with prior results while matching Manuel's interface style.

## Short Oral Update

I reviewed Manuel's model-function file and followed the same interface pattern for my RNN and GRU models. The new functions take train and test dataframes plus the training target, build 24-month sequences internally, train either a vanilla RNN or a GRU, and return predictions with dates. The key settings match my benchmark script: hidden size 24, Adam optimizer, MSE loss, early stopping, and gradient clipping. This makes my RNN/GRU models easier to plug into the team's shared evaluation loop instead of keeping them isolated in a standalone script.
