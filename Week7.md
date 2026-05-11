# goals
benchmark statisctical models

2 quantum papers we are inspired
- quantum enhancement features for signature kernel
- quantum reservoir computing

The use case:
1. for problems where we have a limited amount of data 
where a small increase in prediction accuracy has a huge amplification 
that are time sensitive where training costs are at a premium
2. trying to recapture the advantages of the kernel trick and reservoir computing
perform high dimension transformations on data to revela patterns that are easier to model
perform transformation in hilbert space
 
 Areas:
1. arima, linear regression 
  Jack + the base quantum circuit
2. SVM with signature kernels Vanilla (later we will do quantum)
  David
https://github.com/crispitagorico/sigkernel
https://www.moodys.com/web/en/us/about/what-we-do/quantum-computing/recession-prediction.html

4. RNN, Classical reservoir computing you do not train the weights except for the final layer
  William
5. Get the best models from https://www.tejakonduri.com/uploads/Konduri_JMP.pdf and recreate on our data
   Manuel

(We have the expanding window, but from practical consideratinos of both the reseroir we need a constant window length.)


------------------------------------------------------------------
# goals
1. Everyone on Thursday - Wrap each model in a function that imports a train + test dataframe, then outputs predictions and dates.
1. David - A standard notebook that loops through data tracks, target variables, forecast horizons and poos train test splits. Then saves the performance metrics of y_pred for each of these problems in a standardized format.
2. Jack - Compile list of performance metrics and curated features for each target variable. Because we have relatively little data, I doubt the execution of the actual metrics will be a bottleneck
  - I vote we use this library, and the metrics that are listed in the top bullet point section: https://skforecast.org/latest/user_guides/metrics
  - Mean Squared Error (mean_squared_error)
  - Mean Absolute Error (mean_absolute_error)
  - Mean Absolute Percentage Error (mean_absolute_percentage_error)
  - Symmetric Mean Absolute Percentage Error (symmetric_mean_absolute_percentage_error)
  - Mean Squared Log Error (mean_squared_log_error)
  - Median Absolute Error (median_absolute_error)
  - Mean Absolute Scaled Error (mean_absolute_scaled_error)
  - Root Mean Squared Scaled Error (root_mean_squared_scaled_error)
3. Manuel - Find a source to justify the choice of a curated set of predictors for unemployment and industrial production. Aiming for a small number (6-10), can be the current track b or a different set. The important thing is that we have evidence and
a defendible reason behind it.
4. William - Find a source to justify the choice of a curated set of predictors for CPI and Inflation. Aiming for a small number (6-10). The important thing is that we have evidence and
a defendible reason behind it.


