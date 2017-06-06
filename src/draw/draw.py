# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore")
from pandas.tseries.offsets import Minute, Hour
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def loadData(filename,tollgate_id,direction,day):
    volume1 = pd.read_csv(filename)
    volume1['volume'] = volume1['volume'].astype('float')
    volume1.index = volume1['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    volume1['time_window_week'] = volume1.index.map(lambda x: x.weekday)
    volume1['time_window_month'] = volume1.index.map(lambda x: x.month)
    volume1['time_window_day'] = volume1.index.map(lambda x: x.day)
    volume1['time_window_hour'] = volume1.index.map(lambda x: x.hour)
    volume1['time_window_minute'] = volume1.index.map(lambda x: x.minute)
    volume1.drop(
        pd.date_range(start=datetime(2016, 9, 28), end=datetime(2016, 10, 10), freq='20min', closed='left'),
        inplace=True, errors='ignore')

    volume = volume1.sort_values(
        ['time_window_hour', 'time_window_minute', 'tollgate_id', 'direction'])
    # print volume
    volume = volume[
        ['tollgate_id', 'direction', 'time_window_month', 'time_window_week', 'time_window_day', 'time_window_hour',
         'time_window_minute', 'volume']]
    volume = volume[(volume['time_window_month']==10)&(volume['time_window_day']>=25)&(volume['time_window_day']<=31)]
    tmp = volume[(volume['time_window_month']==10)&(volume['tollgate_id']==tollgate_id)&(volume['direction']==direction)&(volume['time_window_day']==day)]
    # print tmp
    return tmp.values[:,-1]

def draw():
    filename= '../data/processed_2/test2_' + str(20) + 'min_avg_volume.csv'
    filename7 = '../model/answer/ensemble_3model_0.1187.csv'
    filename8 = '../model/answer/result_stack_25-31_extdata.csv'
    filename10 = '../model/answer/result_merge_3.csv'
    filename11 = '../model/answer/2017-05-27ATVNumeric5.csv'
    filename12 = '../model/answer/result_merge_all.csv'
    for tollgate_id in range(1,4):
        for direction in range(2):
            if tollgate_id ==2 and direction ==1:
                continue
            for day in range(25,32):
                plt.figure(figsize=(8, 4))
                tmp1 = loadData(filename,tollgate_id,direction,day)
                tmp2 = loadData(filename8,tollgate_id,direction,day)
                tmp3 = loadData(filename10, tollgate_id, direction, day)
                tmp4 = loadData(filename11, tollgate_id, direction, day)
                tmp5 = loadData(filename7, tollgate_id, direction, day)
                tmp6 = loadData(filename12, tollgate_id, direction, day)
                # print tmp1,tmp2
                ans1 = np.concatenate((tmp1[:6],tmp2[:6]))
                ans2 = np.concatenate((tmp1[6:],tmp2[6:]))
                ans3 = np.concatenate((tmp1[:6], tmp3[:6]))
                ans4 = np.concatenate((tmp1[6:], tmp3[6:]))
                ans5 = np.concatenate((tmp1[:6], tmp4[:6]))
                ans6 = np.concatenate((tmp1[6:], tmp4[6:]))
                ans7 = np.concatenate((tmp1[:6], tmp5[:6]))
                ans8 = np.concatenate((tmp1[6:], tmp5[6:]))
                ans9 = np.concatenate((tmp1[:6], tmp6[:6]))
                ans10 = np.concatenate((tmp1[6:], tmp6[6:]))
                # print ans1,ans2
                plt.plot([i for i in range(1,7)], ans1[:6], label="", color="red", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans1[5:], label="aeoluss", color="green", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans3[5:], label="jie_rain", color="blue", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans5[5:], label="oneday", color="gold", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans7[5:], label="mice", color="lightcoral", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans9[5:], label="ensemble", color="yellowgreen", linewidth=2)
                plt.title(str(tollgate_id)+" "+ str(direction)+" "+str(day)+"_89")
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.savefig('../images/'+str(tollgate_id)+"_"+ str(direction)+"_"+str(day)+"_89.jpg")
                plt.close()

                plt.figure(figsize=(8, 4))
                plt.plot([i for i in range(1,7)], ans2[0:6], label="", color="red", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans2[5:], label="aeoluss", color="green", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans4[5:], label="jie_rain", color="blue", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans6[5:], label="oneday", color="gold", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans8[5:], label="mice", color="lightcoral", linewidth=2)
                plt.plot([i for i in range(6, 13)], ans10[5:], label="ensemble", color="yellowgreen", linewidth=2)
                plt.title(str(tollgate_id)+" "+ str(direction)+" "+str(day)+"_1718")
                plt.legend(loc='upper left')
                plt.grid(True)
                plt.savefig('../images/'+str(tollgate_id)+"_"+ str(direction)+"_"+str(day)+"_1718.jpg")
                plt.close()

if __name__ == '__main__':
    draw()