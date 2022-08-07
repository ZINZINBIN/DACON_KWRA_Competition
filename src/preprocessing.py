import pandas as pd
import glob2
import os
import numpy as np
from tqdm.auto import tqdm
from typing import List

X_COLS = ["swl", "inf", "sfw", "ecpc","tototf", "tide_level","fw_1018662", "fw_1018680","fw_1018683", "fw_1019630"]
Y_COLS = ["wl_1018662", "wl_1018680","wl_1018683", "wl_1019630"]

def merge_all_data(root_dir : str):

    w_data_paths = sorted(glob2.glob(os.path.join(root_dir, "water_data/*.csv")))

    df_total = pd.DataFrame()

    for w_path in w_data_paths:
        df = pd.read_csv(w_path)
        df = df.replace(" ", np.nan)
        df = df.interpolate(method = 'values')
        df = df.fillna(0)

        df_total = pd.concat([df_total, df], axis = 0)
        df_total = df_total.reset_index(drop = True)

    return df_total

def preprocessing(df : pd.DataFrame, seq_len : int, x_cols : List = X_COLS, y_cols : List = Y_COLS):
    data = []
    target = []

    total_size = len(df)

    for idx in tqdm(range(total_size - seq_len)):
        data.append(
            df[x_cols].loc[idx : idx + seq_len].values
        )

        target.append(
            df[y_cols].loc[idx + seq_len].values
        )

    data = np.array(data).reshape(-1, seq_len, len(x_cols))
    target = np.array(target).reshape(-1, len(y_cols))

    print("data : ", data.shape)
    print("target : ", target.shape)

    return data, target