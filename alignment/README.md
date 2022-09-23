# 核验点对齐代码说明


##  1. 模型输入输出说明及相关数据存储位置

该模型解决手写公式的识别问题，模型预测时输入一张手写公式图，输出预测序列。

模型训练时接收
- 手写公式图（image）: 为一组大小不一的图像
- 图像掩模（image_mask）: 统一图像大小，构造dataloader时自动生成
- 标签（label）: 长短不一的预测序列
- 标签（label mask）：统一预测序列长度，用于计算损失

我们从训练数据中抽取了16组数据作为小数据集存放于alignment/lite_data/下

- lite_images.pkl
- lite_labels.txt
- words_dict.txt: 符号字典，用于序列编码

根据上述真实训练数据生成假数据的假数据为(存放于step1-5/data/)
- fake_data.npy: 大小为数据集图像平均值的（1，103, 314）的np数组，共8张
- fake_data_mask.npy: 全1的（1，103, 314）数组
- fake_label.npy: 长度为24的序列
- fake_label_mask.npy: 全1的长度为24的序列

模型训练时输出
- word_probs：输出序列的概率值
- counting_preds: 用于计数损失计算
- counting_preds1: 用于计数损失计算
- counting_preds2: 用于计数损失计算

##  2. 模型结构对齐

该核验点使用假数据，使用torch初始化的参数进行一轮训练后对齐其输出。阈值为1e-5.请运行如下代码：

    python alignment/step1-5/01_test_forward.py

##  3. 数据读取对齐

该检验点使用小数据集，查看数据集读取是否对齐，该步骤关闭shuffle确保读取顺序一直，请运行如下代码：

    python alignment/step1-5/02_test_data.py

##  4. 评估指标对齐

该检验点使用小数据集，对比一轮训练输出的评估指标，阈值为1e-5.请运行如下代码

    python alignment/step1-5/03_test_metric.py

##  5. 损失对齐

该检验点使用假数据，对比一轮训练后的损失是否对齐，阈值为1e-6.请运行如下代码

    python alignment/step1-5/04_test_loss.py

##  6. 反向对齐

该检验点使用小数据集，对比两轮训练的损失，阈值为1e-6.请运行如下代码

    python alignment/step1-5/05_test_backward.py