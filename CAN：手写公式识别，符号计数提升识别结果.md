# CAN：手写公式识别，符号计数提升识别结果

### 1、论文解读

​		Counting-Aware Network（CAN）是2022年ECCV会议收录的手写数学公式识别新算法，其主要创新点是：（1）设计了多尺度计数模块（Multi-Scale Counting Module，MSCM）计数每一个符号的出现次数，从而提升检测的准确率；（2）设计了结合计数的注意力解码器：使用位置编码表征特征图中不同空间位置，加强模型对于空间位置的感知。

（架构图）

### 2、复现详情

#### 2.1 实验数据： CROHME

​		CROHME数据集的内容是一系列手写数学公式图片，共包含8884个样本(其中训练样本8835）。其中图片均为单通道灰度图，为黑色背景白色手写公式，标签（符号序列）是由111类符号组成的不定从长序列，数据集示例如下图所示。

#### 2.2 模型搭建

组网部分

#### 2.3 模型训练

模型优化配置，超参数等

#### 2.4 模型预测

### 3、结果展示

### 4、总结

### 5、参考资料

论文连接：[When Counting Meets HMER: Counting-Aware Network for Handwritten Mathematical Expression Recognition](https://arxiv.org/abs/2207.11463)

原文代码：https://github.com/LBH1024/CAN