__author__ = 'guoyang3'
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def Test2(rootDir):
    file_list = []
    for lists in os.listdir(rootDir):
        path = os.path.join(rootDir, lists)
        file_list.append(path)
        if os.path.isdir(path):
            Test2(path)
    return file_list

path = "result"
file_list = Test2(path)
print file_list
df = pd.read_csv(file_list[0])

file_list.remove(file_list[0])
for x in file_list:
    dftmp = pd.read_csv(x)
    df = df.merge(dftmp, on=["tollgate_id", "time_window", "direction"])

result_list = []
for index, row in df.iterrows():
    volume_list = row[3:].tolist()
    # print volume_list
    volume_list1 = sorted(volume_list)

    result = np.mean([volume_list1[0], volume_list1[1]])
    # result = np.mean(volume_list1)

    result_list.append(result)
print result_list
df = df[["tollgate_id", "time_window", "direction"]]
df["volume"] = result_list
print len(result_list)
print np.mean(result_list)

df["time_window_start"] = pd.to_datetime(df["time_window"])
df["time_window_end"] = df["time_window_start"] + timedelta(minutes=20)
list_tw = []
for x in range(0, len(df["time_window_start"] )):
    str_tw =  '[' + str(df["time_window_start"][x]) + ',' + str(df["time_window_end"][x]) + ')'
    list_tw.append(str_tw)

df["time_window"] = list_tw
df = df[["tollgate_id", "time_window", "direction","volume"]]



path  =  os.path.dirname( os.getcwd())
df.to_csv("../../answer/result_gy.csv", index=False)
# test(df)