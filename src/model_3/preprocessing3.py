__author__ = 'guoyang3'

import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

if __name__ == '__main__':
    path = "../../data/data_after_process/tmp_file/"
    df_train = pd.read_csv(path + "0_train.csv")
    df_train = df_train.append(
        pd.read_csv(path + "5_train.csv")).append(
        pd.read_csv(path + "10_train.csv")).append(
        pd.read_csv(path + "15_train.csv"))

    # print df_train.tail(5)
    df_train["time"] = df_train["time_start"]
    # print df_train.tail(5)
    df_train = df_train.sort("time")
    df_grouped = df_train.groupby(["tollgate_id", "direction"])

    ts_feature = []

    for key, data in df_grouped:
        data = pd.DataFrame.fillna(data, 0)
        data["hour"] = data["time"].apply(lambda x: int(x[11: 13]))
        data["miniute"] = data["time"].apply(lambda x: int(x[14: 16]))
        data["time"] = pd.to_datetime(data["time"])
        data = data.sort("time")
        data = data.set_index("time")

        ts = data["volume"]
        ts = ts.asfreq('5Min', method='pad')
        plt.plot(ts)

        freq = ((24 * 60) // 5)
        decomposition = seasonal_decompose(ts, freq=freq)
        trend = decomposition.trend
        seasonal = decomposition.seasonal
        residual = decomposition.resid
        seasonal.name = 'seasonal'
        residual.name = 'residual'
        trend.name = 'trend'
        data = data[["tollgate_id", "direction","hour","miniute"]]
        data = pd.concat([data, seasonal], axis=1)

        print data.head()
        print len(data)
        data=data.drop_duplicates(['hour','miniute','seasonal'])
        print len(data)
        ts_feature.append(data)
        # break

    df_ts = pd.concat(ts_feature, axis=0)
    df_ts.to_csv("ts_feature2_simple.csv",index=False)


