__author__ = 'guoyang3'

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn import cross_validation


def remove_exception(data):
    data = data[
        (data["date"] != "2016-09-30") | (data["tollgate_id"] != 1) | (data["direction"] != 0) | (data["am_pm"] != 0)]

    data = data[
        (data["date"] != "2016-09-30") | (data["tollgate_id"] != 1) | (data["direction"] != 1) | (
            data["am_pm"] != 0) | (data["period_num"].isin([4, 5, 6]))]

    data = data[
        (data["date"] != "2016-09-21") | (data["tollgate_id"] != 2) | (data["direction"] != 0) | (data["am_pm"] != 1)]
    data = data[
        (data["date"] != "2016-09-24") | (data["tollgate_id"] != 2) | (data["direction"] != 0) | (
            data["am_pm"] != 0) | (data["period_num"].isin([4, 5, 6]))]
    data = data[
        (data["date"] != "2016-09-27") | (data["tollgate_id"] != 2) | (data["direction"] != 0) | (
            data["am_pm"] != 0) | (data["period_num"] == 2)]
    data = data[
        (data["date"] != "2016-09-28") | (data["tollgate_id"] != 2) | (data["direction"] != 0)]
    data = data[
        (data["date"] != "2016-09-30") | (data["tollgate_id"] != 2) | (data["direction"] != 0) | (
            data["am_pm"] != 0) | (data["period_num"].isin([3, 4, 5, 6]))]

    data = data[
        (data["date"] != "2016-09-30") | (data["tollgate_id"] != 3) | (data["direction"] != 0) | (
            data["am_pm"] != 0) | (data["period_num"] == 5)]

    data = data[
        (data["date"] != "2016-09-30") | (data["tollgate_id"] != 3) | (data["direction"] != 1) | (
            data["am_pm"] != 0) | (data["period_num"].isin([4, 5, 6]))]

    return data


def feature_transform_log_rf(key, data):
    data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    data = data.drop("period_num", axis=1)

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['day_of_week'])], axis=1)
    data = data.drop("day_of_week", axis=1)
    data = data.drop("first_last_workday", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        data["direction1"] = data['direction']
    return data


def feature_transform_split(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    # data = data.drop("rel_humidity", axis= 1)




    # data["sum"] = data["0"] + data["1"] + data["2"] + data["3"] + data["4"] + data["5"]

    data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    data = data.drop("period_num", axis=1)

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    data = data.drop("first_last_workday", axis=1)

    data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


def feature_transform_gbrt(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    # data = data.drop("rel_humidity", axis= 1)




    # data["sum"] = data["0"] + data["1"] + data["2"] + data["3"] + data["4"] + data["5"]

    data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    data = data.drop("period_num", axis=1)

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    data = data.drop("first_last_workday", axis=1)

    data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


def feature_transform_knn(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    n = 30
    data["am_pm"] = data["am_pm"] * n
    data["period_num"] = data["period_num"] * n
    data["holiday"] = data["holiday"] * n
    data["first_last_workday"] = data["first_last_workday"] * n
    data["precipitation"] = data["precipitation"] * n
    data['tollgate_id'] = data['tollgate_id'] * n
    data['direction'] = data['direction'] * n
    data['day_of_week'] = data['day_of_week'] * 2

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    data = data.drop("rel_humidity", axis=1)
    # data = data.drop("holiday", axis=1)
    data = data.drop("first_last_workday", axis=1)
    data = data.drop("day_of_week", axis=1)

    if (key == 1):
        data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    return data


def feature_transform_svr(key, data):
    # data = remove_exception(data)

    data["precipitation"] = data[["precipitation"]].fillna(value=0)
    data["rel_humidity"] = data[["rel_humidity"]].fillna(value=50)

    data["precipitation"] = data["precipitation"].apply(lambda x: x > 0)
    # data["rel_humidity"] = data["rel_humidity"].apply(lambda x: x > 90)

    data = data.drop("precipitation", axis=1)
    # data = data.drop("rel_humidity", axis= 1)




    # data["sum"] = data["0"] + data["1"] + data["2"] + data["3"] + data["4"] + data["5"]


    data = pd.concat([data, pd.get_dummies(data['period_num'])], axis=1)
    data = data.drop("period_num", axis=1)

    data = pd.concat([data, pd.get_dummies(data['holiday'])], axis=1)
    data = data.drop("holiday", axis=1)
    #
    # data = pd.concat([data, pd.get_dummies(data['first_last_workday'])], axis=1)
    # data = data.drop("first_last_workday", axis=1)

    if (key == 1):
        data = pd.concat([data, pd.get_dummies(data['tollgate_id'])], axis=1)
        # data["tollgate_id1"] = data['tollgate_id']
        data["direction1"] = data['direction']
    # print normalize(data['0'])
    data['0'] = MinMaxScaler().fit_transform(data['0'])

    data['1'] = MinMaxScaler().fit_transform(data['1'])
    data['2'] = MinMaxScaler().fit_transform(data['2'])
    data['3'] = MinMaxScaler().fit_transform(data['3'])
    data['4'] = MinMaxScaler().fit_transform(data['4'])
    data['5'] = MinMaxScaler().fit_transform(data['5'])
    data['seasonal'] = MinMaxScaler().fit_transform(data['seasonal'])
    data["rel_humidity"] = MinMaxScaler().fit_transform(data['rel_humidity'])

    return data


if __name__ == '__main__':
    pd.set_option('display.width', 1000)

