#-*- coding:utf8 -*-#
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from matplotlib import pyplot as plt
# import seaborn as sns
from dateutil.parser import parse
import warnings
warnings.filterwarnings("ignore")
from pandas.tseries.offsets import Minute, Hour, Day
import numpy as np

BeforeNationlDay = {
    date(2016,9,30)
}

NationalDayStart = {
    date(2016,10,1),
    date(2016,10,2),
    date(2016,10,3),
    date(2016,10,4),
}

NationalDayEnd = {
    date(2016,10,5),
    date(2016,10,6),
    date(2016,10,7)
}

WorkingWeekend = {
    date(2016, 10, 8),
    date(2016, 10, 9)
}

def data_preprocess(volume_file, volume_file_new, test_volume_file):

    volume = pd.read_csv(volume_file)
    volume_new = pd.read_csv(volume_file_new)
    test_volume = pd.read_csv(test_volume_file)

    time_window = pd.date_range(start=datetime(2016,9,19),end=datetime(2016,10,18),freq='20min',closed='left').map(lambda x:'['+str(x)+','+str(x+Minute(20))+')')
    fill_null_dataframe = pd.DataFrame({'tollgate_id':len(time_window)*2*[1]+len(time_window)*[2]+len(time_window)*2*[3],
                                        'direction':len(time_window)*[0]+len(time_window)*[1]+len(time_window)*[0]+len(time_window)*[0]+len(time_window)*[1],
                                        'time_window':np.tile(time_window,5)})
    volume = pd.merge(volume,fill_null_dataframe, how='right').fillna(0).sort_values(['tollgate_id','direction','time_window']) #use fill null values
    volume = pd.concat((volume,volume_new),ignore_index=True)
    volume['volume'] = volume['volume'].astype('float')
    volume['tollgate_id'] = volume['tollgate_id'].astype('int')
    volume['direction'] = volume['direction'].astype('int')
    volume.index = volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    volume['id'] = 'T' + volume.tollgate_id.map(str) + 'D' + volume.direction.map(str)

    test_volume['volume'] = test_volume['volume'].astype('float')
    test_volume['tollgate_id'] = test_volume['tollgate_id'].astype('int')
    test_volume['direction'] = test_volume['direction'].astype('int')
    test_volume.index = test_volume['time_window'].map(lambda x: parse(x.split(',')[0][1:]))
    test_volume['id'] = 'T' + test_volume.tollgate_id.map(str) + 'D' + test_volume.direction.map(str)
    return volume, test_volume

def peak_time(time):
    # peak type
    if 7 <= time.hour <= 9:
        return 'EarlyPeakTime'
    elif 17 <= time.hour <= 20:
        return 'LatePeakTime'
    else:
        return 'NormalTime'

def day_class(date):
    # day type
    dayofweek = date.dayofweek
    date = date.date()
    if date in BeforeNationlDay:
        return 'BeforeNationalDay'
    elif date in NationalDayStart:
        return 'NationalDayStart'
    elif date in NationalDayEnd:
        return 'NationalDayEnd'
    elif date in WorkingWeekend:
        return 'WorkingWeekend'
    elif date in WorkingWeekend:
        return 'WorkingWeek'
    elif dayofweek in range(5):
        return 'WorkingDay'
    else:
        return 'Weekend'

def precipitation_grade(precipitation):
    # precipitation type
    if precipitation == 0.0:
        return 'Sunny'
    if precipitation < 5.0:
        return 'SmallRain'
    if 5.0 <= precipitation < 10.0:
        return 'MediumRain'
    if 10.0 <= precipitation:
        return 'BigRain'

def get_py_datetime(x):
    date_str = str(x[0]) + ' ' + str(x[1])
    return datetime.strptime(date_str,"%Y-%m-%d %H")

def sarimax_preparation_main(volume, test_volume, weather_train_file, weather_train_new_file, weather_test_file):

    train_data = volume.pivot(columns='id', values='volume')
    test_data = test_volume.pivot(columns='id', values='volume')
    tmp = pd.DataFrame({'T1D0':np.NaN,'T1D1':np.NaN,'T2D0':np.NaN,'T3D0':np.NaN,'T3D1':np.NaN},index=pd.date_range(start=datetime(2016,10,18)+Day(7),end=datetime(2016,10,25)+Day(7),freq='20min',closed='left'))
    tmp.set_value(test_data.index,test_data.columns, test_data.values)
    train_data = pd.concat((train_data, tmp))

    peakDummies = pd.get_dummies(train_data.index.map(peak_time))
    peakDummies.index = train_data.index
    dayDummies = pd.get_dummies(train_data.index.map(day_class))
    dayDummies.index = train_data.index
    train_data = pd.concat([train_data,peakDummies, dayDummies],axis=1)

    weather_train = pd.read_csv(weather_train_file)
    weather_train_new = pd.read_csv(weather_train_new_file)
    weather_test = pd.read_csv(weather_test_file)

    weather_full = pd.concat([weather_train, weather_train_new, weather_test]).reset_index()
    weather_full.precipitation = weather_full.precipitation * 4
    weather_full['precipitation'] = weather_full.precipitation.map(
        precipitation_grade)
    rain_dummies = pd.get_dummies(weather_full.precipitation)
    weather_feature = weather_full[['date', 'hour']].join(rain_dummies)
    weather_feature.index = weather_feature.apply(get_py_datetime, axis=1)
    weather_feature = weather_feature.resample('20min').first()
    weather_feature = weather_feature.fillna(method='ffill')
    weather_feature['diff'] = np.abs(weather_feature.index.map(
        lambda dt: dt.hour) - weather_feature.hour)
    weather_feature[weather_feature['diff'] >= 3] = np.nan
    del weather_feature['diff'],weather_feature['date'],weather_feature['hour']
    train_data = train_data.join(weather_feature)
    train_data.to_csv('sarimax_data.csv',index=True)

def main():
    basepath = '../../data/'
    volume_file = basepath + 'data_after_process/training_20min_avg_volume.csv'
    volume_file_new = basepath + 'data_after_process/training2_20min_avg_volume.csv'
    test_volume_file = basepath + 'data_after_process/test2_20min_avg_volume.csv'
    weather_train_file = basepath + 'dataset/weather (table 7)_training_update.csv'
    weather_train_new_file = basepath + 'dataset/weather (table 7)_test1.csv'
    weather_test_file = basepath + 'dataset/weather (table 7)_2.csv'
    volume, test_volume = data_preprocess(volume_file, volume_file_new, test_volume_file)
    print 'The data of sarimax is preparating!'
    sarimax_preparation_main(volume, test_volume, weather_train_file, weather_train_new_file, weather_test_file)

if __name__ == '__main__':
    main()