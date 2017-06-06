__author__ = 'guoyang3'

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import numpy as np
from feature_transform import *
from sklearn.neighbors import KNeighborsRegressor
# from test import test

def main():
    df_train0 = pd.read_csv("train.csv")
    df_train1 = pd.read_csv("train1.csv")
    df_train2 = pd.read_csv("train2.csv")
    df_train3 = pd.read_csv("train3.csv")
    df_train_list = [df_train0,
                     df_train1, df_train2, df_train3
    ]
    df_train = pd.concat(df_train_list)
    len_train = len(df_train)

    df_test = pd.read_csv("test2.csv")
    df_train = df_train.append(df_test)[df_train.columns.tolist()]

    df_date = pd.read_csv("date.csv")
    df_ts = pd.read_csv("ts_feature2_simple.csv")

    df_train = df_train.merge(df_date, on="date", how="left")

    df_train = df_train.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")

    data = pd.DataFrame.reset_index(df_train)
    data = data.drop("index", axis=1)

    data = feature_transform_log_rf(key=1, data= data)

    y = data.ix[:len_train - 1]["volume"]
    x = data.ix[:len_train - 1, 8:]
    x1 = data.ix[len_train:, 8:]
    y = np.log(y)

    regressor_cubic = RandomForestRegressor(n_estimators=700, max_features='sqrt', random_state=10, oob_score=True)

    regressor_cubic.fit(x, y)

    yhat = regressor_cubic.predict(x1)
    yhat = np.power(np.e, yhat)
    df_test["volume"] = yhat
    df_test = df_test[['tollgate_id', 'time_window', 'direction', 'volume']]
    df_test.to_csv("result/result_rf_"+str(np.mean(yhat))+".csv", index=False)

    print np.mean(yhat)
    print regressor_cubic.oob_score_


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    main()