# -*- coding:utf8 -*-#
import sys
sys.path.append('..')
import feature as feature
import params as params
import predict as predict
import test as test

def main():
    feature.main()
    '''
        利用网格搜索选择参数，时间较长，可以不运行
    '''
    # params.main()

    '''
        利用最后一星期的数据来测试模型的效果
    '''
    test.main()

    '''
        预测结果
    '''
    predict.main()

if __name__ == '__main__':
    main()