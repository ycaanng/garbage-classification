# 天池Apache Flink极客挑战赛——垃圾图片分类

## 成绩

season  |   rank  |  score
  :-:   |   :-:   |   :-:
season1 | 12/2684 |  94.50
season2 | 8/2684  |  74.90

## 数据来源

1. 数据地址：链接：<https://pan.baidu.com/s/15teD9cvpwx4qiCOIba0IQw> 提取码：wblw
2. 官方java包：链接：<https://pan.baidu.com/s/1ZwGLywgzq1IpKuMklo_jnw> 提取码：97m0
3. garbage-origin：9930张
4. garbage-official：300张
5. saved_model：模型参数

## 环境要求

1. garbage-classification-python中的脚本在windows、linux、mac下皆可使用，需要的python package都是常见的包
2. garbage-classification-java中工程仅能在ubuntu环境中执行，本人环境为ubuntu18.04+python3.6.8，需要下载的python包为networkx、tensorflow-1.14.0(版本号必须，gpu、cpu版本不限)、numpy

## 数据解释

1. garbage-origin：图片来源于网络下载
2. garbage-official：图片来源于官方视频截图
3. saved_model：

- inception-resnet-v2-tf文件夹可以在python中利用tf.compat.v1.keras.experimental.load_from_saved_model()方法加载，并用于预测；也可以在java中利用intel-anlytics-zoo加载并预测
- inception-resnet-v2-tf.weights.h5仅可以在python中利用model.load_weights()方法加载

## 本地自测使用方法

1. 根据前面给出的url下载数据，放到指定位置
2. 根据前面给出的url下载官方java包，安装到本地的maven仓库
3. 利用garbage-classification-python中的model_pretrain_v1.py训练并评估结果
4. 利用garbage-classification-java中的OfflineRunMain.java预测
5. 在使用garbage-classification-java中的OfflineRunMain.java时，需要传入IMAGE_INPUT_PATH、IMAGE_MODEL_PATH

- 一种方法是在系统的环境变量中加入这两个值，利用export IMAGE_INPUT_PATH=""和export IMAGE_MODEL_PATH=""
- 另一种方法是，在IDEA的OfflineRunMain.java的Configuration中的Environment variables中加入这两个参数

## 比赛使用方法

1. 根据前面给出的url下载数据，放到指定位置
2. 根据前面给出的url下载官方java包，安装到本地的maven仓库
3. 本地利用garbage-classification-python中的model_pretrain_v2.py训练并评估结果
4. 线上利用garbage-classification-python中的model.py加载本地预训练模型再训练
5. 线上利用garbage-classification-java中的OnlineRunMainSeason2.java预测图片

## 结果

1. 将garbage-origin数据集按8：2划分为训练集和测试集，训练出来的模型在测试集的准确率为0.82+，该模型在garbage-official上的准确率为0.60+，最好能达到0.65+
2. 将1训练的模型保存后，在官方线上数据集中再训练，最终模型的准确率为0.749
