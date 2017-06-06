# -*- coding:utf8 -*-#
import sys
sys.path.append('..')
import data_processing.aggregate_volume as aggregate_volume
import data_processing.data_processing as data_processing
import model_1.run as model1_run
import model_2.run as model2_run
import model_3.run as model3_run
import model_4.run as model4_run

def main():
    '''
        生成每20分钟统计值的文件
    '''
    aggregate_volume.main()
    '''
        数据预处理，处理国庆节，同时将数据划分为上下午的test和train
    '''
    data_processing.main()

    model1_run.main()
    model2_run.main()
    model3_run.main()
    model4_run.main()

'''
run 所有的结果
'''

if __name__ == '__main__':
    main()