# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from dateutil.parser import parse

'''
两个结果权重融合
'''
def ensemble_2(filename1, filename2, ratio, fileto):
    fr1 = open(filename1)
    fr2 = open(filename2)
    fr_to = open(fileto, "w")
    flag = 0
    for line1, line2 in zip(fr1.readlines(), fr2.readlines()):
        if flag == 0:
            fr_to.write(line1)
            flag = 1
            continue
        data1 = line1.strip().split(",")
        data2 = line2.strip().split(",")
        num1 = float(data1[-1])
        num2 = float(data2[-1])
        num3 = num1 * ratio + num2 * (1 - ratio)
        fr_to.write(",".join(data1[:-1]) + "," + str(num3) + "\n")
    fr_to.close()

'''
三个模型权重融合
'''
def ensemble_3(filename1, filename2, filename3, ratio1, ratio2, ratio3, fileto):
    fr1 = open(filename1)
    fr2 = open(filename2)
    fr3 = open(filename3)
    fr_to = open(fileto, "w")
    flag = 0
    for line1, line2, line3 in zip(fr1.readlines(), fr2.readlines(), fr3.readlines()):
        if flag == 0:
            fr_to.write(line1)
            flag = 1
            continue
        data1 = line1.strip().split(",")
        data2 = line2.strip().split(",")
        data3 = line3.strip().split(",")
        num1 = float(data1[-1])
        num2 = float(data2[-1])
        num3 = float(data3[-1])
        num = num1 * ratio1 + num2 * ratio2 + num3 * ratio3
        fr_to.write(",".join(data1[:-1]) + "," + str(num) + "\n")
    fr_to.close()

'''
计算两个结果的绝对值距离
'''
def get_distance(filename1, filename2):
    fr1 = open(filename1)
    fr2 = open(filename2)
    ans = 0.0
    num = 0
    flag = 0
    for line1, line2 in zip(fr1.readlines(), fr2.readlines()):
        if flag == 0:
            flag = 1
            continue
        data1 = line1.strip().split(",")
        data2 = line2.strip().split(",")
        num1 = float(data1[-1])
        num2 = float(data2[-1])
        ans += math.fabs(num1 - num2)
        num += 1
    return ans / num

'''
统一格式进行排序
'''
def sortAnsFile(ans_file):
    volume = pd.read_csv(ans_file)
    volume['volume'] = volume['volume'].astype('float')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    # print volume
    tmp = volume[['tollgate_id', 'time_window', 'direction', 'volume']].sort_values(
        ['tollgate_id', 'direction', 'time_window'])
    tmp.to_csv(ans_file, float_format='%.5f', header=True, index=False,
               encoding='utf-8')

if __name__ == '__main__':
    filename1 = '../../answer/predict_2_mean.csv'
    filename2 = '../../answer/knn_sarimax_ensemble.csv'
    sortAnsFile(filename1)
    sortAnsFile(filename2)
    ratio = 0.7
    fileto1 = '../../answer/result_model1_model2_0.7.csv'
    ensemble_2(filename1, filename2, ratio, fileto1)

    list = [filename1, filename2, fileto1]
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            scored = get_distance(list[i], list[j])
            print str(i + 1) + " " + str(j + 1), scored
