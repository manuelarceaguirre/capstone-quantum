import numpy as np
import pandas as pd
import warnings
from pmdarima import auto_arima


def arima_model_function(train_df, test_df, y_train, horizon):
    """
    Fits Auto-ARIMA once on training data and forecasts horizon steps ahead.
    No recursive refitting.
    """
    y = y_train.shift(horizon - 1).dropna()
    y.index = pd.to_datetime(y.index) 
    y = y.asfreq('MS')
    # --- single fit ---
    model=auto_arima(y,start_p=0,d=1,start_q=0,
        max_p=5,max_d=5,max_q=5, start_P=0,
        D=1, start_Q=0, max_P=5,max_D=5,
        max_Q=5, m=12, seasonal=False,
        error_action='warn',
        supress_warnings=True,
        random_state=20,n_fits=50)

    # --- direct multi-step forecast ---
    n_periods=(horizon+len(test_df))
    prediction = pd.DataFrame(model.predict(n_periods = n_periods),index=test_df.index)

    return prediction.iloc[-len(test_df):, 0]