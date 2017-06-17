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
两个结果权重融合
'''


def ensemble_2_NN_rf(filename1, filename2, ratio, fileto):
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
        flag += 1
        if (flag >= 230) and (flag <= 253):
            fr_to.write(",".join(data1) + "\n")
        else:
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

def addHand(filename):
    data=[]
    fr = open(filename)
    for line in fr.readlines():
        if line[0]=='t':
            return False
        data.append(line.strip())
    fr=open(filename,"w")
    fr.write("tollgate_id,time_window,direction,volume\n")
    for line in data:
        fr.write(line+"\n")
    fr.close()


def sortAnsFile(ans_file):
    volume = pd.read_csv(ans_file)
    volume['volume'] = volume['volume'].astype('float')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    # print volume
    tmp = volume[['tollgate_id', 'time_window', 'direction', 'volume']].sort_values(
        ['tollgate_id', 'direction', 'time_window'])
    tmp.to_csv(ans_file, float_format='%.5f', header=True, index=False,
               encoding='utf-8')


def special_judge(filename1, filename2, fileto, tollgate_id, direction, day, mark=0):
    volume1 = pd.read_csv(filename1)
    volume1['volume'] = volume1['volume'].astype('float')
    volume1.index = volume1['time_window'].map(lambda x: parse(x.split(',')[0][1:]))

    volume2 = pd.read_csv(filename2)
    volume2['volume'] = volume2['volume'].astype('float')
    volume2.index = volume2['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    if mark == 0:
        volume2_special = volume2[(volume2['tollgate_id'] == tollgate_id) & (volume2['direction'] == direction)][
            day].between_time('8:00', '9:40')
    else:
        volume2_special = volume2[(volume2['tollgate_id'] == tollgate_id) & (volume2['direction'] == direction)][
            day].between_time('17:00', '18:40')

    def replace(x):
        if x['tollgate_id'] == tollgate_id and x['direction'] == direction and x.name in volume2_special.index:
            return volume2_special.loc[x.name, 'volume']
        else:
            return x['volume']

    volume1['volume'] = volume1.apply(replace, axis=1)

    tmp = volume1[['tollgate_id', 'time_window', 'direction', 'volume']].sort_values(
        ['tollgate_id', 'direction', 'time_window'])

    tmp.to_csv(fileto, float_format='%.5f', header=True, index=False,
               encoding='utf-8')


if __name__ == '__main__':
    filename1 = './answer/predict_2_mean.csv'
    filename2 = './answer/linear_sarimax_ensemble.csv'
    filename3 = './answer/result_gy.csv'
    filename4 = './answer/ATVNumeric5.csv'
    addHand(filename4)
    filename5 = './answer/predict_rice_1.csv'
    filename6 = './answer/predict_rice_final.csv'
    sortAnsFile(filename1)
    sortAnsFile(filename2)
    sortAnsFile(filename3)
    sortAnsFile(filename4)
    sortAnsFile(filename5)
    sortAnsFile(filename6)

    ratio = 0.7
    fileto1 = './answer/result_model1_model2_0.7_0.3.csv'
    ensemble_2(filename2, filename1, ratio, fileto1)

    ratio = 0.5
    fileto2 = './answer/result_NN_model3_0.5_0.5.csv'
    ensemble_2_NN_rf(filename3, filename4, ratio, fileto2)

    ratio1 = 0.45
    ratio2 = 0.45
    ratio3 = 0.1
    fileto3 = './answer/result_ensemble_0.45_0.45_0.1.csv'
    ensemble_3(filename6, fileto1, fileto2, ratio1, ratio2, ratio3, fileto3)

    ratio1 = 0.4
    ratio2 = 0.35
    ratio3 = 0.25
    fileto4 = './answer/result_ensemble_0.40_0.35_0.25.csv'
    ensemble_3(filename6, fileto1, fileto2, ratio1, ratio2, ratio3, fileto4)

    # 1-0 2016-10-30 8:00-10:00  rice_1
    fileto5 = './answer/result_final_submit_result.csv'
    special_judge(fileto3, filename5, fileto5, 1, 0, day='2016-10-30', mark=0)

    # 2-0 2016-10-31 8:00-10:00  7_3
    special_judge(fileto5, fileto1, fileto5, 2, 0, day='2016-10-31', mark=0)

    # 2-0 2016-10-31 17:00-18:00  40_35_25
    special_judge(fileto5, fileto4, fileto5, 2, 0, day='2016-10-31', mark=1)

    # 2-0 2016-10-30 17:00-18:00  aeoluss
    special_judge(fileto5, filename3, fileto5, 2, 0, day='2016-10-30', mark=1)


    # filename7 = './answer/result_5_31_merge_all_0.45_0.45_0.1_change.csv'
    list = [fileto1,fileto2, filename6,fileto5]
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            scored = get_distance(list[i], list[j])
            print str(i + 1) + " " + str(j + 1), scored
