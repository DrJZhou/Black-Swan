__author__ = 'guoyang3'

import pandas as pd


def ampm(x):
    if (x <= 12 ):
        return 1
    return 0


def calc_period_num(x):
    if x == 17 or x == 8:
        return 1
    return 4


def generate_train(filname):
    df_volume = pd.read_csv(filname)

    df_volume["time"] = pd.to_datetime(df_volume["time"])
    df_volume = df_volume.sort(['tollgate_id', 'direction', 'time'])
    df_volume["am_pm"] = df_volume["hour"].apply(lambda x: ampm(x))

    for shift_num in range(0, 6):
        f2 = lambda x: x.values[shift_num]

        df_volume[str(shift_num)] = df_volume[["tollgate_id", "direction", "volume", "date", "am_pm"]].groupby(
            ["tollgate_id", "direction", "date", "am_pm"]).transform(f2)

    df_volume = df_volume[
        (df_volume["hour"] == 8) |
        (df_volume["hour"] == 9) |
        (df_volume["hour"] == 17) |
        (df_volume["hour"] == 18)]

    df_volume["period_num"] = df_volume["hour"].apply(lambda x: calc_period_num(x))
    df_volume["period_num"] = df_volume["period_num"] + df_volume["miniute"].apply(lambda x: x / 20)

    df_volume["hour1"] = df_volume["hour"].apply(lambda x: x / 3 * 3)
    df_weather = pd.read_csv("../../data/data_after_process/tmp_file/weather_feature.csv")[["date", "hour1", "precipitation", "rel_humidity"]]
    df_volume = df_volume.merge(df_weather, on=["date", "hour1"], how="left")

    df_volume = df_volume.drop("hour1", axis=1)


    return df_volume



def combined_train():
    path = "../../data/data_after_process/tmp_file/"
    df1 = generate_train(path+"0_train_filter.csv")
    print df1.shape
    df1["volume"] = df1["volume"].replace(0, 1)
    df1.to_csv("train.csv", index=False)

    df2 = generate_train(path+"5_train_filter.csv")
    print df2.shape
    df2["volume"] = df2["volume"].replace(0, 1)
    df2.to_csv("train1.csv", index=False)

    df3 = generate_train(path+"10_train_filter.csv")
    print df3.shape
    df3["volume"] = df3["volume"].replace(0, 1)
    df3.to_csv("train2.csv", index=False)

    df4 = generate_train(path+"15_train_filter.csv")
    print df4.shape
    df4["volume"] = df4["volume"].replace(0, 1)
    df4.to_csv("train3.csv", index=False)


def get_test():
    path = "../../data/data_after_process/tmp_file/"
    df1 = generate_train(path+"0_test_filter.csv")
    df1.to_csv("test2.csv", index=False)


def main():

    combined_train()
    get_test()


if __name__ == '__main__':
    pd.set_option('display.width', 1000)
    main()