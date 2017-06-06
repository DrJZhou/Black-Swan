__author__ = 'guoyang3'
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from feature_transform import *
import numpy as np
from sklearn.grid_search import GridSearchCV
import random
# from test import test

def main():
    df_test = pd.read_csv("test2.csv")

    df_train0 = pd.read_csv("train.csv")
    df_train1 = pd.read_csv("train1.csv")
    df_train2 = pd.read_csv("train2.csv")
    df_train3 = pd.read_csv("train3.csv")
    df_train_list = [df_train0,
                     df_train1, df_train2, df_train3
    ]
    random.shuffle(df_train_list)
    df_train = pd.concat(df_train_list)

    df_ts = pd.read_csv("ts_feature2_simple.csv")
    df_date = pd.read_csv("date.csv")

    df_train = df_train.merge(df_date, on="date", how="left")
    df_train = df_train.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")

    df_test = df_test.merge(df_date, on="date", how="left")
    df_test = df_test.merge(df_ts, on=["tollgate_id", "hour", "miniute", "direction"], how="left")

    df_train_grouped = df_train.groupby(["tollgate_id", "direction"])
    df_test_grouped = df_test.groupby(["tollgate_id", "direction"])
    result = []
    oob = []
    for key, train_data in df_train_grouped:

        test_data = df_test_grouped.get_group(key)
        len_train = len(train_data)
        train_data = train_data.append(test_data)[train_data.columns.tolist()]
        train_data = feature_transform_split(key, train_data)


        regressor_cubic = RandomForestRegressor(n_estimators=500, max_features='sqrt', random_state=10, oob_score=True)

        train_data = pd.DataFrame.reset_index(train_data)
        train_data = train_data.drop("index", axis=1)
        y = train_data.ix[:len_train - 1, :]["volume"]


        x = train_data.ix[:len_train - 1, 8:]
        x1 = train_data.ix[len_train:, 8:]

        regressor_cubic.fit(x, y)
        yhat = regressor_cubic.predict(x1)

        test_data["volume"] = yhat
        result.append(test_data[['tollgate_id', 'time_window', 'direction', 'volume']])


    df_result = pd.concat(result, axis=0)

    df_result.to_csv("result/result_split_rf"+str(np.mean(df_result["volume"]))+".csv", index=False)




if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    main()