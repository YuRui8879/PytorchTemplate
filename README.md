#### 介绍

编写了一个pytorch框架的模板，方便以后直接使用。参考了很多设计，最终使用这种框架，主要包括了三个部分，算法，模型及数据加载器。其他的类都是用于辅助训练的，目录文件也是按此组织的

#### 目录

-root  
|-Algorithm  
| |-EarlyStopping   提前停止类，当结果无提高时提前结束训练  
| |-Regularization  正则化类，用于向权重及偏置添加L1或L2正则化项  
| |-Tester          测试器，测试训练好的模型  
| |-Trainer         训练器，训练模型  
|-DataLoader   
| |-OfflineLoader   离线数据读取器，将所有数据加载至内存后开始训练   
| |-OnlineLoader    在线数据读取器，加载部分数据至内存后开始训练，实现边加载边训练  
|-Metrics  
| |-Metrics         评价指标类，向该类添加评价指标，训练时会自动测试模型性能  
| |-Classification  分类指标，已实现的部分分类指标  
| |-Regression      回归指标，已实现的部分回归指标  
|-Model  
| |-Block           一些用于加强模型的模块，例如注意力模块  
| |-VGG             已实现的VGG模型  
main                程序入口，目前还是面向过程的写法，以后会进行封装  

#### 如何使用该模板

需要实现离线数据读取器或在线数据读取器的数据读取方法，此外，在main.py中设置相关的超参数及损失函数，优化器等

#### 目前已实现（已收集）的模型

* LeNet
* AlexNet
* VGG
* ResNet
* SENet
* DenseNet


