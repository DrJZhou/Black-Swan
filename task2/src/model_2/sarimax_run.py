#-*- coding:utf-8 -*-#
import sarimax_preparation
import sarimax_model
import sarimax_predict

def main():

    sarimax_preparation.main()
    sarimax_model.sarimax_model_build()
    sarimax_predict.sarimax_predict()

'''
时间序列模型
'''

if __name__ == '__main__':
    main()