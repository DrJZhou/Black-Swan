# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
import math
from datetime import datetime, timedelta
from dateutil.parser import parse


def merge(filename1, filename2, ratio, fileto):
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
        data2 = line2.strip()
        num1 = float(data1[-1])
        num2 = float(data2)
        num3 = num1 * ratio + num2 * (1 - ratio)
        fr_to.write(",".join(data1[:-1]) + "," + str(num3) + "\n")
    fr_to.close()


def merge_2(filename1, filename2, ratio, fileto):
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


def merge_3(filename1, filename2, filename3, ratio1, ratio2, ratio3, fileto):
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


def get_scoring(filename1, filename2):
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


def sortAnsFile(ans_file):
    volume = pd.read_csv(ans_file)
    volume['volume'] = volume['volume'].astype('float')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    # print volume
    tmp = volume[['tollgate_id', 'time_window', 'direction', 'volume']].sort_values(
        ['tollgate_id', 'direction', 'time_window'])
    tmp.to_csv(ans_file, float_format='%.5f', header=True, index=False,
               encoding='utf-8')


def testHalf(filename, fileto1, fileto2):
    fr = open(filename)
    fr_to1 = open(fileto1, 'w')
    fr_to2 = open(fileto2, 'w')
    flag = 0
    for line1 in fr.readlines():
        if flag == 0:
            fr_to1.write(line1)
            fr_to2.write(line1)
            flag += 1
            continue
        data1 = line1.strip().split(",")
        num1 = float(data1[-1])
        if flag <= 210:
            fr_to1.write(",".join(data1[:-1]) + "," + str(num1) + "\n")
            fr_to2.write(",".join(data1[:-1]) + "," + str(0.00) + "\n")
        else:
            fr_to1.write(",".join(data1[:-1]) + "," + str(0.00) + "\n")
            fr_to2.write(",".join(data1[:-1]) + "," + str(num1) + "\n")
        flag += 1
    fr_to1.close()
    fr_to2.close()


if __name__ == '__main__':
    filename1 = 'answer/predict_2017-05-26_2_mean.csv'
    filename4 = 'answer/prediction_2017-05-25_15_30_knn_no_split_new.csv'
    filename5 = 'answer/prediction_2017-05-27_12_38_0.8_avg_offset.csv'
    filename6 = 'answer/prediction_2017-05-26_10_07_sarimax.csv'
    filename7 = 'answer/ensemble_3model_0.1187.csv'
    filename8 = 'answer/result_stack_25-31_extdata.csv'
    filename9 = 'answer/2017-05-27ATVNumeric5_2_0.csv'
    # sortAnsFile(filename9)

    filename11 = 'answer/separate_tollgate_direction_lightgbm_01154.csv'
    filename12 = 'answer/2017-05-29basicNumeric8.csv'
    filename13 = 'answer/rice01443.csv'
    filename14 = 'answer/prediction_2017-05-25_15_30_knn_no_split_new_and_sarimax_merge.csv'
    sortAnsFile(filename13)
    sortAnsFile(filename14)
    ratio = 0.7
    fileto4 = 'answer/result_merge_1494_1595_7_3.csv'
    merge_2(filename14, filename1, ratio, fileto4)

    ratio = 0.5
    fileto6 = 'answer/result_oneday_aeoluss.csv'
    merge_2(filename8, filename9, ratio, fileto6)

    ratio1 = 0.45
    ratio2 = 0.45
    ratio3 = 0.1
    fileto2 = 'answer/result_5_31_merge_all_0.45_0.45_0.1.csv'
    merge_3(filename13, fileto4, fileto6, ratio1, ratio2, ratio3, fileto2)

    list = [filename1, filename14, filename4, filename5, filename6, filename7, filename8, filename9,
            fileto6, filename14, fileto4]
    for i in range(len(list)):
        for j in range(i + 1, len(list)):
            scored = get_scoring(list[i], list[j])
            print str(i + 1) + " " + str(j + 1), scored
