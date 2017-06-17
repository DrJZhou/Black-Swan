执行 run.py     最后结果（result_gy.csv）输出到上一级目录下

脚本目录
gy

数据来源
data/dataset/

======数据预处理========

generate_20min_volume.py    
统计每20分钟车流量； 4种时间窗口  （后移 0,5,10,15 min）

preprocessing0.py
车流量数据预处理，根据小时过滤， 只保留6,7,8,9，15,16,17,18 点数据

preprocessing1.py
天气数据预处理

preprocessing2.py
合并天气和车流量，生成部分特征

preprocessing3.py
时间序列分解，得到seasonal特征

preprocessing4.py
日期特征，节假日，首个\最后工作日 ， 每周第n天


======模型========

model1.py
随机森林
使用额外时间窗口数据
每个 tollgate\direction  训练一个模型

model2.py
knn
不使用额外时间窗口数据
每个 tollgate\direction  训练一个模型

model3.py
随机森林
过滤异常值
使用额外时间窗口数据
volume = log（volume）
所有tollgate\direction  训练一个模型

model4.py
GBRT
使用额外时间窗口数据
所有tollgate\direction  训练一个模型

model5.py
knn
使用额外时间窗口数据
所有tollgate\direction  训练一个模型


======融合========
stacking.py
融合5个模型结果
将最后结果（result_gy.csv）输出到上一级目录下
