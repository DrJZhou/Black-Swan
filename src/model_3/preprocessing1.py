__author__ = 'guoyang3'
import pandas as pd
import os

def main():
    path = "../../data/dataset/"
    df_weather = pd.read_csv(path+"weather (table 7)_training_update.csv")
    df_weather = df_weather.append(
        pd.read_csv(path+"weather (table 7)_test1.csv")).append(
        pd.read_csv(path+"weather (table 7)_2.csv")
    )

    print df_weather.head()

    print df_weather.head()
    df_weather["hour1"] = df_weather["hour"]
    df_weather[
        ["date", "hour1", "pressure", "sea_pressure", "wind_direction", "wind_speed", "temperature", "rel_humidity",
         "precipitation"]].to_csv("../../data/data_after_process/tmp_file/weather_feature.csv", index=False)



if __name__ == '__main__':
    pd.set_option('display.width', 1000)

    main()