__author__ = 'guoyang3'

import  pandas as pd

# generate date information
# col  'holiday'    0: workday   1: weekend   2: holiday
# col   'first_last_workday'     1: first workday of week  2: last workday of week  0: other
# col  'day_of_week'     1: Mon  2: Tues    ...  7 : Sun


df_date = pd.DataFrame(columns=('date', 'holiday', 'first_last_workday','day_of_week'))
df_date.loc[0] = ['2016-09-19', '0', '0', '1']
df_date.loc[1] = ['2016-09-20', '0', '0', '2']
df_date.loc[2] = ['2016-09-21', '0', '0', '3']
df_date.loc[3] = ['2016-09-22', '0', '0', '4']
df_date.loc[4] = ['2016-09-23', '0', '2', '5']
df_date.loc[5] = ['2016-09-24', '1', '0', '6']
df_date.loc[6] = ['2016-09-25', '1', '0', '7']
df_date.loc[7] = ['2016-09-26', '0', '1', '1']
df_date.loc[8] = ['2016-09-27', '0', '0', '2']
df_date.loc[9] = ['2016-09-28', '0', '0', '3']
df_date.loc[10] = ['2016-09-29', '0', '0', '4']
df_date.loc[11] = ['2016-09-30', '0', '2', '5']
df_date.loc[12] = ['2016-10-01', '2', '0', '6']
df_date.loc[13] = ['2016-10-02', '2', '0', '7']
df_date.loc[14] = ['2016-10-03', '2', '0', '1']
df_date.loc[15] = ['2016-10-04', '2', '0', '2']
df_date.loc[16] = ['2016-10-05', '2', '0', '3']
df_date.loc[17] = ['2016-10-06', '2', '0', '4']
df_date.loc[18] = ['2016-10-07', '2', '0', '5']
df_date.loc[19] = ['2016-10-08', '0', '1', '6']
df_date.loc[20] = ['2016-10-09', '0', '0', '7']
df_date.loc[21] = ['2016-10-10', '0', '0', '1']
df_date.loc[22] = ['2016-10-11', '0', '0', '2']
df_date.loc[23] = ['2016-10-12', '0', '0', '3']
df_date.loc[24] = ['2016-10-13', '0', '0', '4']
df_date.loc[25] = ['2016-10-14', '0', '2', '5']
df_date.loc[26] = ['2016-10-15', '1', '0', '6']
df_date.loc[27] = ['2016-10-16', '1', '0', '7']
df_date.loc[28] = ['2016-10-17', '0', '1', '1']
df_date.loc[29] = ['2016-10-18', '0', '0', '2']
df_date.loc[30] = ['2016-10-19', '0', '0', '3']
df_date.loc[31] = ['2016-10-20', '0', '0', '4']
df_date.loc[32] = ['2016-10-21', '0', '2', '5']
df_date.loc[33] = ['2016-10-22', '1', '0', '6']
df_date.loc[34] = ['2016-10-23', '1', '0', '7']
df_date.loc[35] = ['2016-10-24', '0', '1', '1']
df_date.loc[36] = ['2016-10-25', '0', '0', '2']
df_date.loc[37] = ['2016-10-26', '0', '0', '3']
df_date.loc[38] = ['2016-10-27', '0', '0', '4']
df_date.loc[39] = ['2016-10-28', '0', '2', '5']
df_date.loc[40] = ['2016-10-29', '1', '0', '6']
df_date.loc[41] = ['2016-10-30', '1', '0', '7']
df_date.loc[42] = ['2016-10-31', '0', '1', '1']
df_date.loc[43] = ['2016-11-01', '0', '0', '2']
print df_date
df_date.to_csv("date.csv",index=False)