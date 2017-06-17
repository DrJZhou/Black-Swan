import pandas as pd
import numpy as np

def df_filter(df_volume):
    df_volume["time"] = df_volume["time_start"]
    df_volume["date"] = df_volume["time"].apply(lambda x: pd.to_datetime(x[: 10]))
    df_volume["hour"] = df_volume["time"].apply(lambda x: int(x[11: 13]))
    df_volume["miniute"] = df_volume["time"].apply(lambda x: int(x[14: 16]))
    df_volume = df_volume[
        (df_volume["hour"] == 6) | (df_volume["hour"] == 7) |
        (df_volume["hour"] == 8) | (df_volume["hour"] == 9) |
        (df_volume["hour"] == 15) | (df_volume["hour"] == 16) |
        (df_volume["hour"] == 17) | (df_volume["hour"] == 18)]
    df_volume["time_window"] = df_volume["time"]
    df_volume = df_volume[["tollgate_id", "time_window", "direction", "volume", "time", "date", "hour", "miniute"]]
    return  df_volume



def main():
    path = "../../data/data_after_process/tmp_file/"
    df_test = pd.read_csv(path+"0_test.csv")
    df_train1 = pd.read_csv(path+"0_train.csv")
    df_train2 = pd.read_csv(path+"5_train.csv")
    df_train3 = pd.read_csv(path+"10_train.csv")
    df_train4 = pd.read_csv(path+"15_train.csv")

    df_filter(df_test).to_csv(path+"0_test_filter.csv",index=False)
    df_filter(df_train1).to_csv(path+"0_train_filter.csv",index=False)
    df_filter(df_train2).to_csv(path+"5_train_filter.csv",index=False)
    df_filter(df_train3).to_csv(path+"10_train_filter.csv",index=False)
    df_filter(df_train4).to_csv(path+"15_train_filter.csv",index=False)


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    main()
