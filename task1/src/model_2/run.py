#-*- coding:utf-8 -*-#
import linear_regression
import sarimax_run
import linear_sarimax_ensemble
def main():
    '''
    knn model
    由于模型训练参数速度较慢，大约2个小时，这里可以选择是否选择重新跑参数，默认为加载原有的参数
    predict_from_params = False  则表示重新跑参数
    '''
    # knn.main(predict_from_params=False)
    linear_regression.main()

    '''
    sarimax 时间序列模型
    需要三个小时跑模型
    '''
    sarimax_run.main()

    '''
    两个模型结果融合
    '''

    linear_sarimax_ensemble.linear_sarimax_ensemble()

if __name__ == '__main__':
    main()