# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore")
from datetime import datetime, timedelta

def data_processing_train(jiange=20):
    volume = pd.read_csv('../../data/data_after_process/training2_' + str(jiange) + 'min_avg_volume_all.csv')
    volume['volume'] = volume['volume'].astype('float')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    volume['time_window_week'] = volume.index.map(lambda x: x.weekday)
    volume['time_window_month'] = volume.index.map(lambda x: x.month)
    volume['time_window_day'] = volume.index.map(lambda x: x.day)
    volume['time_window_hour'] = volume.index.map(lambda x: x.hour)
    volume['time_window_minute'] = volume.index.map(lambda x: x.minute)
    # volume.drop(
    #     pd.date_range(start=datetime(2016, 10, 1), end=datetime(2016, 10, 8), freq='10min', closed='left'),
    #     inplace=True, errors='ignore')

    # print volume
    volume_need = volume[
        ['tollgate_id', 'direction', 'time_window_month', 'time_window_week', 'time_window_day', 'time_window_hour',
         'time_window_minute', 'volume']].values
    dataset = {}
    data_week = {}
    for line in volume_need:
        tollagate_id = int(line[0])
        direction = int(line[1])
        month = int(line[2])
        week = int(line[3])
        day = int(line[4])
        hour = int(line[5])
        minute = int(line[6])
        volume_value = int(line[7])
        # print tollagate_id,volume_value
        if dataset.get(tollagate_id, 0) == 0:
            dataset[tollagate_id] = {}
            data_week[tollagate_id] = {}
        if dataset[tollagate_id].get(direction, 0) == 0:
            dataset[tollagate_id][direction] = {}
            data_week[tollagate_id][direction] = {}
        if dataset[tollagate_id][direction].get(month, 0) == 0:
            dataset[tollagate_id][direction][month] = {}
            data_week[tollagate_id][direction][month] = {}
        if dataset[tollagate_id][direction][month].get(day, 0) == 0:
            dataset[tollagate_id][direction][month][day] = {}
        data_week[tollagate_id][direction][month][day] = week
        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
            dataset[tollagate_id][direction][month][day][hour] = {}
        dataset[tollagate_id][direction][month][day][hour][minute] = volume_value
        # dataset[tollagate_id][direction][day]['week']=week

    fileto1 = '../../data/data_after_process/train_6_7_' + str(jiange) + '.csv'
    fr_to1 = open(fileto1, "w")
    fileto2 = '../../data/data_after_process/train_15_16_' + str(jiange) + '.csv'
    fr_to2 = open(fileto2, "w")

    fileto3 = '../../data/data_after_process/train_8_9_' + str(jiange) + '.csv'
    fr_to3 = open(fileto3, "w")
    fileto4 = '../../data/data_after_process/train_17_18_' + str(jiange) + '.csv'
    fr_to4 = open(fileto4, "w")
    for tollagate_id in dataset.keys():
        for direction in dataset[tollagate_id].keys():
            start = datetime(2016, 9, 19)
            for i in range(36):
                now = start + timedelta(days=i)
                month = now.month
                day = now.day
                if data_week[tollagate_id][direction][month].get(day, -1) == -1:
                    continue
                week = data_week[tollagate_id][direction][month][day]
                tmp = str(tollagate_id) + "," + str(direction) + "," + str(month) + "," + str(day) + "," + str(week)
                for hour in [6, 7]:
                    for minute in [i * jiange for i in range(60 / jiange)]:
                        # print tmp, minute
                        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
                            volume_value = 0
                        else:
                            volume_value = dataset[tollagate_id][direction][month][day][hour].get(minute, 0)
                        tmp += "," + str(volume_value)
                fr_to1.write(tmp + "\n")

                tmp = str(tollagate_id) + "," + str(direction) + "," + str(month) + "," + str(day) + "," + str(week)
                for hour in [15, 16]:
                    for minute in [i * jiange for i in range(60 / jiange)]:
                        # print tmp, minute
                        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
                            volume_value = 0
                        else:
                            volume_value = dataset[tollagate_id][direction][month][day][hour].get(minute, 0)
                        tmp += "," + str(volume_value)
                fr_to2.write(tmp + "\n")

                tmp = str(tollagate_id) + "," + str(direction) + "," + str(month) + "," + str(day) + "," + str(week)
                for hour in [8, 9]:
                    for minute in [i * jiange for i in range(60 / jiange)]:
                        # print tmp, minute
                        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
                            volume_value = 0
                        else:
                            volume_value = dataset[tollagate_id][direction][month][day][hour].get(minute, 0)
                        tmp += "," + str(volume_value)
                fr_to3.write(tmp + "\n")

                tmp = str(tollagate_id) + "," + str(direction) + "," + str(month) + "," + str(day) + "," + str(week)
                for hour in [17, 18]:
                    for minute in [i * jiange for i in range(60 / jiange)]:
                        # print tmp, minute
                        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
                            volume_value = 0
                        else:
                            volume_value = dataset[tollagate_id][direction][month][day][hour].get(minute, 0)
                        tmp += "," + str(volume_value)
                fr_to4.write(tmp + "\n")

    fr_to1.close()
    fr_to2.close()
    fr_to3.close()
    fr_to4.close()

def data_processing_test(jiange=20):
    volume = pd.read_csv('../../data/data_after_process/test2_' + str(jiange) + 'min_avg_volume.csv')
    volume['volume'] = volume['volume'].astype('float')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    volume['time_window_week'] = volume.index.map(lambda x: x.weekday)
    volume['time_window_month'] = volume.index.map(lambda x: x.month)
    volume['time_window_day'] = volume.index.map(lambda x: x.day)
    volume['time_window_hour'] = volume.index.map(lambda x: x.hour)
    volume['time_window_minute'] = volume.index.map(lambda x: x.minute)
    volume_need = volume[
        ['tollgate_id', 'direction', 'time_window_month', 'time_window_week', 'time_window_day', 'time_window_hour',
         'time_window_minute', 'volume']].values
    dataset = {}
    data_week = {}
    for line in volume_need:
        tollagate_id = int(line[0])
        direction = int(line[1])
        month = int(line[2])
        week = int(line[3])
        day = int(line[4])
        hour = int(line[5])
        minute = int(line[6])
        volume_value = int(line[7])
        # print tollagate_id,volume_value
        if dataset.get(tollagate_id, 0) == 0:
            dataset[tollagate_id] = {}
            data_week[tollagate_id] = {}
        if dataset[tollagate_id].get(direction, 0) == 0:
            dataset[tollagate_id][direction] = {}
            data_week[tollagate_id][direction] = {}
        if dataset[tollagate_id][direction].get(month, 0) == 0:
            dataset[tollagate_id][direction][month] = {}
            data_week[tollagate_id][direction][month] = {}
        if dataset[tollagate_id][direction][month].get(day, 0) == 0:
            dataset[tollagate_id][direction][month][day] = {}
        data_week[tollagate_id][direction][month][day] = week
        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
            dataset[tollagate_id][direction][month][day][hour] = {}
        dataset[tollagate_id][direction][month][day][hour][minute] = volume_value
        # dataset[tollagate_id][direction][day]['week']=week

    fileto1 = '../../data/data_after_process/test_6_7_' + str(jiange) + '.csv'
    fr_to1 = open(fileto1, "w")
    fileto2 = '../../data/data_after_process/test_15_16_' + str(jiange) + '.csv'
    fr_to2 = open(fileto2, "w")
    for tollagate_id in dataset.keys():
        for direction in dataset[tollagate_id].keys():
            start = datetime(2016, 10, 25)
            for i in range(7):
                now = start + timedelta(days=i)
                month = now.month
                day = now.day
                week = data_week[tollagate_id][direction][month][day]
                tmp = str(tollagate_id) + "," + str(direction) + "," + str(month) + "," + str(day) + "," + str(week)
                for hour in [6, 7]:
                    for minute in [i * jiange for i in range(60 / jiange)]:
                        # print minute
                        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
                            volume_value = 0
                        else:
                            volume_value = dataset[tollagate_id][direction][month][day][hour].get(minute, 0)
                        tmp += "," + str(volume_value)
                fr_to1.write(tmp + "\n")

                tmp = str(tollagate_id) + "," + str(direction) + "," + str(month) + "," + str(day) + "," + str(week)
                for hour in [15, 16]:
                    for minute in [i * jiange for i in range(60 / jiange)]:
                        # print minute
                        if dataset[tollagate_id][direction][month][day].get(hour, 0) == 0:
                            volume_value = 0
                        else:
                            volume_value = dataset[tollagate_id][direction][month][day][hour].get(minute, 0)
                        tmp += "," + str(volume_value)
                fr_to2.write(tmp + "\n")
    fr_to1.close()
    fr_to2.close()

def main():
    data_processing_train(20)
    data_processing_test(20)

'''
生成训练集和测试集
1.过滤掉10-1到10-7号的数据
2.将数据分为上午和下午
3.每一天都是一行，如果上午每一行包含了tollagate_id,direction,month,day,week,以及67两个小时每20风中的6个值
4.上午生成train_67,test_67，train_89,里面保存的就是89两个小时每六个小时的值，下午类似
'''

if __name__ == '__main__':
    main()
