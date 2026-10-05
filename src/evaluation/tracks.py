"""Feature tracks: loading and per-fold train/test preparation.

Track config
------------
``track_config[track_name][target] = {"type": ..., "path": ...}`` where type is

- ``"dataframe"``: a parquet of features indexed by date, sliced per fold.
  Precomputed quantum/classical reservoir tracks are of this type.
- ``"pipeline"``: a pickled, unfitted sklearn pipeline. Each fold fits it on
  the training slice of ``stationary_panel`` only, then transforms train and test.

Use :func:`shared_track` when every target uses the same file.
"""
import pickle

import pandas as pd


def shared_track(track_type, path, targets):
    """Config entry for a track whose file is the same for every target."""
    return {target: {"type": track_type, "path": path} for target in targets}


def load_tracks(track_config, active_tracks, targets):
    """
    Load the files for every active (track, target) pair.

    Returns
    -------
    track_data : dict — track_data[track][target] is a DataFrame or unfitted pipeline
    """
    track_data = {}
    for track_name in active_tracks:
        track_data[track_name] = {}
        for target in targets:
            config = track_config[track_name][target]
            if config["type"] == "dataframe":
                obj = pd.read_parquet(config["path"])
                desc = f"shape={obj.shape}"
            elif config["type"] == "pipeline":
                with open(config["path"], "rb") as f:
                    obj = pickle.load(f)
                desc = f"pipeline={type(obj).__name__}"
            else:
                raise ValueError(f"Unknown track type: {config['type']}")
            track_data[track_name][target] = obj
            print(f"  Track {track_name:<8} target {target:<10} loaded   {desc}")
    return track_data


def prepare_track(track_name, target, fold, track_config, track_data, stationary_panel):
    """
    Slice and prepare train/test DataFrames for a given track, target and fold.

    For dataframe tracks: slice directly by date.
    For pipeline tracks: slice stationary panel by date,
        fit pipeline on train only, transform both, return DataFrames.

    Returns
    -------
    train_df : pd.DataFrame
    test_df  : pd.DataFrame
    """
    config = track_config[track_name][target]
    obj = track_data[track_name][target]

    if config["type"] == "dataframe":
        train_df = obj.loc[fold["train_start"]:fold["train_end"]]
        test_df  = obj.loc[fold["test_start"]:fold["test_end"]]

    elif config["type"] == "pipeline":
        train_raw = stationary_panel.loc[fold["train_start"]:fold["train_end"]]
        test_raw  = stationary_panel.loc[fold["test_start"]:fold["test_end"]]

        # fit on train only — no leakage
        pipeline = pickle.loads(pickle.dumps(obj))  # fresh copy
        pipeline.fit(train_raw)

        train_transformed = pipeline.transform(train_raw)
        test_transformed  = pipeline.transform(test_raw)

        cols = [f"f{i}" for i in range(train_transformed.shape[1])]
        train_df = pd.DataFrame(train_transformed, index=train_raw.index, columns=cols)
        test_df  = pd.DataFrame(test_transformed,  index=test_raw.index,  columns=cols)

    return train_df, test_df
