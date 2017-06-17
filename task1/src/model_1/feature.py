# -*- coding:utf8 -*-#
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
# from matplotlib import pyplot as plt
from dateutil.parser import parse
import warnings

warnings.filterwarnings("ignore")
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR, NuSVR
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, ShuffleSplit
from sklearn.metrics import make_scorer
from pandas.tseries.offsets import Minute, Hour
from sklearn.preprocessing import PolynomialFeatures

OHE = OneHotEncoder(sparse=False)
#节假日
festival_day = ['2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']
#是否工作第一天，而且不是周一
isFirstWork_day = ['2016-10-08']
#不是工作第一天但是是周一
isNotFistWork_day = ['2016-09-19', '2016-10-10']
#不是工作前一天，而且不是周日
isBeforeFirstWork_day = ['2016-10-07']
#不是工作日前一天，而且是周六
isNotBeforeFirstWork_day = ['2016-10-02', '2016-10-09']
#是否上班最后一天，而且不是周五
isEndWork_day = []
#是否不是上班最后一天，而且是周五
isNotEndWork_day = ['2016-10-07']
#是否是放假第一天，而不是周六
isAfterEndWork_day = []
#不是放假第一天，但是是周六
isNotAfterEndWork_day = ['2016-10-08']
#是否工作日，而且是周末
isWorkDay = ['2016-10-08', '2016-10-09']
#不是工作日，但是是周一到周五
isNotWorkDay = ['2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']
weekStart = ['2016-09-13', '2016-09-20', '2016-09-27', '2016-10-04', '2016-10-11']
weekEnd = ['2016-09-19', '2016-09-26', '2016-10-03', '2016-10-10', '2016-10-17']

TheDayNotNeed = ['2016-10-01', '2016-10-02', '2016-10-03', '2016-10-04', '2016-10-05', '2016-10-06', '2016-10-07']#

# tollgateId  变成 onehot
def tollgateId_2_onehot(data):
    tall_onehot = OHE.fit_transform(data)
    # print tall_onehot
    return tall_onehot


# 星期几变成onehot
def week_2_onehot(data):
    week_onehot = OHE.fit_transform(data)
    # print week_onehot
    return week_onehot


# 多项式展开
def transform_pf(data, degree=2):
    PF = PolynomialFeatures(degree=degree)
    pf = PF.fit_transform(data)
    # print pf.shape
    return pf


# 计算基本的统计量  max min sum std mean median
def cal_static(data):
    # print data, np.max(data, axis=1).reshape(-1, 1)
    data_max = np.max(data, axis=1).reshape(-1, 1)
    data_min = np.min(data, axis=1).reshape(-1, 1)
    data_sum = np.sum(data, axis=1).reshape(-1, 1)
    data_std = np.std(data, axis=1).reshape(-1, 1)
    data_mean = np.mean(data, axis=1).reshape(-1, 1)
    data_median = np.median(data, axis=1).reshape(-1, 1)
    return data_max, data_min, data_sum, data_std, data_mean, data_median


# 获取方向
def get_direction(data):
    return data


# 计算三个20分钟的sum和avg
def cal_hour(data):
    hour_sum = []
    hour_mean = []
    hour_median = []
    hour_max = []
    hour_min = []
    hour_std = []
    for line in data:
        hour_sum.append([np.sum(line[:3]), np.sum(line[3:])])
        hour_mean.append([np.mean(line[:3]), np.mean(line[3:])])
        hour_median.append([np.median(line[:3]), np.median(line[3:])])
        hour_max.append([np.max(line[:3]), np.max(line[3:])])
        hour_min.append([np.min(line[:3]), np.min(line[3:])])
        hour_std.append([np.std(line[:3]), np.std(line[3:])])

    hour_sum = np.array(hour_sum)
    hour_mean = np.array(hour_mean)
    hour_median = np.array(hour_median)
    hour_max = np.array(hour_max)
    hour_min = np.array(hour_min)
    hour_std = np.array(hour_std)
    return hour_sum, hour_mean, hour_median, hour_max, hour_min, hour_std


# 计算三个10分钟的sum和avg
def cal_hour_10(data):
    hour_sum = []
    hour_mean = []
    hour_median = []
    hour_max = []
    hour_min = []
    hour_std = []
    for line in data:
        hour_sum.append([np.sum(line[:6]), np.sum(line[6:])])
        hour_mean.append([np.mean(line[:6]), np.mean(line[6:])])
        hour_median.append([np.median(line[:6]), np.median(line[6:])])
        hour_max.append([np.max(line[:6]), np.max(line[6:])])
        hour_min.append([np.min(line[:6]), np.min(line[6:])])
        hour_std.append([np.std(line[:6]), np.std(line[6:])])

    hour_sum = np.array(hour_sum)
    hour_mean = np.array(hour_mean)
    hour_median = np.array(hour_median)
    hour_max = np.array(hour_max)
    hour_min = np.array(hour_min)
    hour_std = np.array(hour_std)
    return hour_sum, hour_mean, hour_median, hour_max, hour_min, hour_std


#计算残差
def cal_trend(data):
    return data[:,1:]-data[:,:5]

# 加入十分钟的特征
def get_10_minute_feature(filename):
    a = np.loadtxt(filename, delimiter=",")
    a = data_fliter(a)
    ten_mintue_feature = a[:, -12:]
    train_max, train_min, train_sum, train_std, train_mean, train_median = cal_static(a[:, -12:])
    ten_mintue_feature = np.hstack((ten_mintue_feature, train_max))
    ten_mintue_feature = np.hstack((ten_mintue_feature, train_min))
    ten_mintue_feature = np.hstack((ten_mintue_feature, train_sum))
    ten_mintue_feature = np.hstack((ten_mintue_feature, train_std))
    ten_mintue_feature = np.hstack((ten_mintue_feature, train_mean))
    ten_mintue_feature = np.hstack((ten_mintue_feature, train_median))

    hour_sum, hour_mean, hour_median, hour_max, hour_min, hour_std = cal_hour_10(a[:, -12:])
    ten_mintue_feature = np.hstack((ten_mintue_feature, hour_sum))
    ten_mintue_feature = np.hstack((ten_mintue_feature, hour_mean))
    ten_mintue_feature = np.hstack((ten_mintue_feature, hour_median))
    ten_mintue_feature = np.hstack((ten_mintue_feature, hour_max))
    ten_mintue_feature = np.hstack((ten_mintue_feature, hour_min))
    ten_mintue_feature = np.hstack((ten_mintue_feature, hour_std))

    pf = transform_pf(hour_mean)
    ten_mintue_feature = np.hstack((ten_mintue_feature, pf))
    return ten_mintue_feature


# 人体舒适度指数：SSD=(1.818t+18.18)(0.88+0.002f)+(t-32)/(45-t)-3.2v+18.2 其中：温度t，湿度f，风速v
def cal_SSD(t, f, v):
    SSD = (1.818 * t + 18.18) * (0.88 + 0.002 * f) + (t - 32) / (45 - t) - 3.2 * v + 18.2
    return SSD


# 加入天气特征
def weather(filename, data, isAfternoon=0):
    volume = pd.read_csv(filename)
    volume['hour'] = volume['hour'].astype('int')
    volume['pressure'] = volume['pressure'].astype('float')
    volume['wind_speed'] = volume['wind_speed'].astype('float')
    volume['temperature'] = volume['temperature'].astype('float')
    volume['rel_humidity'] = volume['rel_humidity'].astype('float')
    volume['precipitation'] = volume['precipitation'].astype('float')
    # print volume
    ans1 = []
    ans2 = []
    ans3 = []
    for line in data:
        month = int(line[2])
        day = int(line[3])
        month_day = "2016-%02d-%02d" % (month, day)
        # print month_day, datetime(2016, month, day)
        if isAfternoon == 0:
            tmp = volume[(volume['date'] == month_day) & (volume['hour'] > 3) & (volume['hour'] < 9)][
                [ 'precipitation']].values
        else:
            tmp = volume[(volume['date'] == month_day) & (volume['hour'] > 12) & (volume['hour'] < 18)][
                [ 'precipitation']].values
        tmp = tmp.flatten()
        if len(tmp)==0:
            # print -1
            ans1.append(ans1[-1])
            continue

        if tmp[0] < 0.01:
            ans1.append([0])
        elif tmp[0] < 5:
            ans1.append([1])
        else:
            ans1.append([2])

    ans1=np.array(ans1)
    # print ans1,ans2,ans3
    # ans1 = OHE.fit_transform(ans1)
    # print ans1
    # print ans1.shape
    # ans2 = OHE.fit_transform(ans2)
    # ans3 = OHE.fit_transform(ans3)
    # ans2 = np.hstack((ans1, ans2))
    # print ans2.shape
    # ans3 = np.hstack((ans2, ans3))
    # print ans3.shape
    return ans1

# 判断特殊时间
def isSpecial(data):
    ans = []
    for line in data:
        month = int(line[0])
        day = int(line[1])
        week = int(line[2])
        month_day = "2016-%02d-%02d" % (month, day)
        # print month_day
        tmp = []

        # 判断是否工作日
        if month_day in isWorkDay:
            tmp.append(1)
        elif month_day in isNotWorkDay:
            tmp.append(0)
        elif week >= 0 and week < 5:
            tmp.append(1)
        else:
            tmp.append(0)

        # 判断是否第一天工作
        if month_day in isFirstWork_day:
            tmp.append(1)
        elif month_day in isNotFistWork_day:
            tmp.append(0)
        elif week == 0:
            tmp.append(1)
        else:
            tmp.append(0)

        # 判断是否工作前一天
        if month_day in isBeforeFirstWork_day:
            tmp.append(1)
        elif month_day in isNotBeforeFirstWork_day:
            tmp.append(0)
        elif week == 6:
            tmp.append(1)
        else:
            tmp.append(0)

        # 判断是否工作最后一天
        if month_day in isEndWork_day:
            tmp.append(1)
        elif month_day in isNotEndWork_day:
            tmp.append(0)
        elif week == 4:
            tmp.append(1)
        else:
            tmp.append(0)

        # 判断是否放假第一天
        if month_day in isAfterEndWork_day:
            tmp.append(1)
        elif month_day in isNotAfterEndWork_day:
            tmp.append(0)
        elif week == 5:
            tmp.append(1)
        else:
            tmp.append(0)

        # 判断是否节假日
        if month_day in festival_day:
            tmp.append(1)
        else:
            tmp.append(0)

        # 判断是否周末
        if week < 5:
            tmp.append(0)
        else:
            tmp.append(1)

        ans.append(tmp)
    ans = np.array(ans)
    # print ans.shape
    return ans


# 数据过滤，去除异常的天
def data_fliter(data):
    ans = []
    for line in data:
        id = int(line[0])
        direction = int(line[1])
        month = int(line[2])
        day = int(line[3])
        # tmp1=[id,direction,month,day]
        # if tmp1 in DonotNeed:
        #     continue
        month_day = "2016-%02d-%02d" % (month, day)
        if month_day in TheDayNotNeed:
            continue
        ans.append(line)
    ans = np.array(ans)
    # print data.shape, ans.shape
    return ans


# 计算和均值比例
def cal_ratio(data):
    dataset = {}
    for line in data:
        id = int(line[0])
        direction = int(line[1])
        month = int(line[2])
        day = int(line[3])
        # if not ((month==10 and day>=11 and day<=17) or (month==9 and day>=20 and day<=26)):
        #     continue
        week = int(line[4])
        if dataset.get(id, 0) == 0:
            dataset[id] = {}
        if dataset[id].get(direction, 0) == 0:
            dataset[id][direction] = {}
        if dataset[id][direction].get(week, 0) == 0:
            dataset[id][direction][week] = []
        # print month,day,line
        dataset[id][direction][week].append(line[-6:])
    all_ratio = {}
    for id in [1, 2, 3]:
        all_ratio[id] = {}
        for direction in [0, 1]:
            if dataset[id].get(direction, 0) == 0:
                continue
            all_ratio[id][direction] = {}
            for week in [i for i in range(7)]:
                tmp = np.array(dataset[id][direction][week])
                # print tmp
                tmp = np.sum(tmp, axis=0)
                ratio1 = tmp[:3] / np.mean(tmp[:3])
                ratio2 = tmp[3:] / np.mean(tmp[3:])
                # ratio1 = tmp[:3] / np.median(tmp[:3],axis=0)
                # ratio2 = tmp[3:] / np.median(tmp[3:],axis=0)
                ratio = np.hstack((ratio1, ratio2))
                all_ratio[id][direction][week] = ratio
    return all_ratio

def scoring(y_true, y_pred):
    return np.sum((y_true - y_pred) * (y_true - y_pred))
    # return np.mean(np.abs((y_true - y_pred) / y_true))

# 得到y
def cal_y(data):
    ans = []
    for line in data:
        ans.append([np.mean(line[:3]), np.mean(line[3:])])
    ans = np.array(ans)
    # print ans
    return ans


# 保存特征
def save_feature_label(filename, x_file):
    a = np.loadtxt(filename, delimiter=",")
    a = data_fliter(a)
    # train_X=np.array([])
    '''
    tollgateId特征onehot
    '''
    tollgateId = tollgateId_2_onehot(a[:, 0].reshape((-1, 1)))
    train_X = tollgateId
    # train_X = a[:, 0].reshape((-1, 1))
    '''
        direction特征onehot
    '''
    direction = get_direction(a[:, 1].reshape((-1, 1)))
    train_X = np.hstack((train_X, direction))
    # train_X = np.hstack((train_X, a[:, 1].reshape((-1, 1))))

    '''
    week 周几onehot特征
    '''
    week = week_2_onehot((a[:, 4].reshape((-1, 1))))
    train_X = np.hstack((train_X, week))
    # train_X = np.hstack((train_X, a[:, 4].reshape((-1, 1))))
    '''
    原始每20分钟的值
    '''
    train_X = np.hstack((train_X, a[:, -6:]))

    '''
    原始一个小时内3个值的max，mean，median，max，min，std等统计量
    '''
    hour_sum, hour_mean, hour_median, hour_max, hour_min, hour_std = cal_hour(a[:, -6:])
    train_X = np.hstack((train_X, hour_sum))
    train_X = np.hstack((train_X, hour_mean))
    train_X = np.hstack((train_X, hour_median))
    train_X = np.hstack((train_X, hour_max))
    train_X = np.hstack((train_X, hour_min))
    train_X = np.hstack((train_X, hour_std))

    '''
    两个小时的均值的二项式
    '''
    pf = transform_pf(hour_mean)
    train_X = np.hstack((train_X, pf))


    '''
    原始2个小时内6个值的max，mean，median，max，min，std等统计量
    '''
    train_max, train_min, train_sum, train_std, train_mean, train_median = cal_static(a[:, -6:])
    train_X = np.hstack((train_X, train_max))
    train_X = np.hstack((train_X, train_min))
    train_X = np.hstack((train_X, train_sum))
    train_X = np.hstack((train_X, train_std))
    train_X = np.hstack((train_X, train_mean))
    train_X = np.hstack((train_X, train_median))

    '''
    是否特殊天，比如工作日，工作第一天，放假第一天，上班前一天，是否节假日等
    '''

    spcial_day = isSpecial(a[:, 2:5])
    train_X = np.hstack((train_X, spcial_day))

    # trend=cal_trend(a[:,-6:])
    # train_X = np.hstack((train_X, trend))
    # print trend.shape,train_X.shape


    print train_X.shape

    np.savetxt(x_file, train_X, delimiter=",")

# 保存train_y
def save_label_ratio(filename, y_file, ratio_file,train_x_filename,train_y_filename,test_x_filename):
    import cPickle
    a = np.loadtxt(filename, delimiter=",")
    a = data_fliter(a)
    # print a
    ratio = cal_ratio(a)
    # ratio = cal_ratio_knn(train_x_filename,train_y_filename,test_x_filename,k)
    cPickle.dump(ratio, open(ratio_file, "wb"))
    train_Y = cal_y(a[:, -6:])
    np.savetxt(y_file, train_Y, delimiter=",")

def main():
    jiange = 20

    '''
    下午的数据特征提取，train_feature和test_feature以及train_y，算ratio
    '''
    afterNoon = "15_16"
    afterNoon_ = "17_18"
    filename = '../../data/data_after_process/train_' + afterNoon + '_' + str(jiange) + '.csv'
    x_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    morning_file = '../../data/data_after_process/train_6_7_' + str(jiange) + '.csv'
    save_feature_label(filename, x_file)

    filename = '../../data/data_after_process/test_' + afterNoon + '_' + str(jiange) + '.csv'
    x_file = '../../data/data_after_process/train_test/test_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    morning_file = '../../data/data_after_process/test_6_7_' + str(jiange) + '.csv'
    save_feature_label(filename, x_file)

    filename1 = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    y_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    ratio_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_ratio.pkl'

    train_x_filename = '../../data/data_after_process/train_' + afterNoon + '_' + str(jiange) + '.csv'
    train_y_filename = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    test_x_filename = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    save_label_ratio(filename1, y_file, ratio_file, train_x_filename, train_y_filename, test_x_filename)

    '''
        上午的数据特征提取，train_feature和test_feature以及train_y，算ratio
    '''
    afterNoon = "6_7"
    afterNoon_ = "8_9"
    filename = '../../data/data_after_process/train_' + afterNoon + '_' + str(jiange) + '.csv'
    x_file = '../../data/data_after_process/train_test/train_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    save_feature_label(filename, x_file)

    filename = '../../data/data_after_process/test_' + afterNoon + '_' + str(jiange) + '.csv'
    x_file = '../../data/data_after_process/train_test/test_' + afterNoon + '_' + str(jiange) + '_feature.csv'
    save_feature_label(filename, x_file)

    filename1 = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    y_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_y.csv'
    ratio_file = '../../data/data_after_process/train_test/train_' + afterNoon_ + '_' + str(jiange) + '_ratio.pkl'

    train_x_filename = '../../data/data_after_process/train_' + afterNoon + '_' + str(jiange) + '.csv'
    train_y_filename = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    test_x_filename = '../../data/data_after_process/train_' + afterNoon_ + '_' + str(jiange) + '.csv'
    save_label_ratio(filename1, y_file, ratio_file, train_x_filename, train_y_filename, test_x_filename)


'''
这文件是用来提取特征的，并保存到文件中，将处理后的数据分为feature和y，同时计算每一个20分钟相对于该小时的平均值的比例
特征：tollgateId特征onehot，direction特征onehot，week 周几onehot特征，原始每20分钟的值，原始一个小时内3个值的max，mean，median，max，min，std等统计量，
两个小时的均值的二项式，原始2个小时内6个值的max，mean，median，max，min，std等统计量，是否特殊天，比如工作日，工作第一天，放假第一天，上班前一天，是否节假日等
算比例的方法：
    比如算周一8：00-8：20相对于8：00-9：00之间车流量的比例：算出历史中每一个周一8：00-8：20的值求一个平均值，算和每一个周一8：00：9：00的均值的比例
'''


if __name__ == '__main__':
    main()