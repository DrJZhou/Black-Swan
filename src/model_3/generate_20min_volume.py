__author__ = 'guoyang3'

import pandas as pd
import os

def run(df,rng):
    rng_length = len(rng)
    result_dfs = []
    for this_direction in range(2):
        for this_tollgate_id in range(1, 4):
            time_start_list = []
            volume_list = []
            direction_list = []
            tollgate_id_list = []

            this_df = df[(df.tollgate_id == this_tollgate_id) & (df.direction == this_direction)]
            if len(this_df) > 0:
                for ind in range(rng_length - 1):
                    this_df_time_window = this_df[(this_df.time >= rng[ind]) & (this_df.time < rng[ind + 1])]
                    volume_list.append(len(this_df_time_window))

                    time_start_list.append(rng[ind])

                result_df = pd.DataFrame({'time_start': time_start_list,
                                          'volume': volume_list,
                                          'direction': [this_direction] * (rng_length - 1),
                                          'tollgate_id': [this_tollgate_id] * (rng_length - 1),
                }
                )
                result_dfs.append(result_df)

    d = pd.concat(result_dfs)

    if type == 'test':
        d['hour'] = d['time_start'].apply(lambda x: x.hour)
        dd = d[d.hour.isin([6, 7, 15, 16])]
    return d




path = "../../data/dataset/"

df_train1 = pd.read_csv(path+"volume(table 6)_training.csv", parse_dates=['time'])
df_train2 = pd.read_csv(path+"volume(table 6)_training2.csv", parse_dates=['date_time'])
df_train2 = df_train2.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})
df_train = df_train1.append(df_train2)

df_test = pd.read_csv(path+"volume(table 6)_test2.csv", parse_dates=['date_time'])
df_test = df_test.rename(columns = {'date_time':'time','tollgate':'tollgate_id','is_etc':'has_etc','veh_type':'vehicle_type','model':'vehicle_model'})

freq = "20min"

# move time window 0 5 10 15 minute
rng1 = pd.date_range("2016-09-19 00:00:00", "2016-10-25 00:00:00", freq=freq)
rng2 = pd.date_range("2016-09-19 00:05:00", "2016-10-25 00:00:00", freq=freq)
rng3 = pd.date_range("2016-09-19 00:10:00", "2016-10-25 00:00:00", freq=freq)
rng4 = pd.date_range("2016-09-19 00:15:00", "2016-10-25 00:00:00", freq=freq)

rng5 = pd.date_range("2016-10-25 00:00:00", "2016-11-01 00:00:00", freq=freq)

run(df_train,rng1).to_csv("../../data/data_after_process/tmp_file/0_train.csv",index= False)
run(df_train,rng2).to_csv("../../data/data_after_process/tmp_file/5_train.csv",index= False)
run(df_train,rng3).to_csv("../../data/data_after_process/tmp_file/10_train.csv",index= False)
run(df_train,rng4).to_csv("../../data/data_after_process/tmp_file/15_train.csv",index= False)
run(df_test,rng5).to_csv("../../data/data_after_process/tmp_file/0_test.csv",index= False)


