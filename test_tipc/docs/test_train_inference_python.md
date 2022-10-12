# Linux GPU/CPU 基础训练推理测试

Linux GPU/CPU 基础训练推理测试的主程序为`test_train_inference_python.sh`，可以测试基于Python的模型训练、评估、推理等基本功能。

## 1. 测试结论汇总

- 训练相关：

| 算法名称 | 模型名称 | 单机单卡 | 单机多卡 |
|  :----: |   :----:  |    :----:  |  :----:   |
|  CAN  | can    | 正常训练    | 正常训练 |


- 推理相关：

| 算法名称 | 模型名称 | device_CPU | device_GPU | batchsize |
|  :----:   |  :----: |   :----:   |  :----:  |   :----:   |
|  CAN   |  can |  支持 | 支持 | 1 |


## 2. 测试流程

### 2.1 准备数据和模型

用于基础训练推理测试的数据位于`/test_images`文件夹内。

```
test_image
    |--lite_data                            # 训练小数据集
    |--test_data                            # 验证小数据集     
    |--test_example                         # 推理使用图片
    |----words_dict.txt                     # 训练、验证、推理所需词表
```
下载用于基础训练推理测试的模型，包括用于验证的预训练模型和用于推理的模型。

```shell
bash test_tipc/prepare.sh ./test_tipc/configs/can/train_infer_python.txt 'lite_train_lite_infer'
```
用于验证的预训练模型会自动放置在`CAN_Paddle`根目录中，用于推理的模型会自动放置在`/test_model`文件夹内。
### 2.2 准备环境


- 安装PaddlePaddle：如果您已经安装了2.2或者以上版本的paddlepaddle，那么无需运行下面的命令安装paddlepaddle。
    ```
    # 需要安装2.2及以上版本的Paddle
    # 安装GPU版本的Paddle
    pip install paddlepaddle-gpu==2.2.0
    # 安装CPU版本的Paddle
    pip install paddlepaddle==2.2.0
    ```

- 安装依赖
    ```
    pip install  -r requirements.txt
    ```
- 安装AutoLog（规范化日志输出工具）
    ```
    pip install  https://paddleocr.bj.bcebos.com/libs/auto_log-1.2.0-py3-none-any.whl
    ```

### 2.3 功能测试


Linux GPU/CPU 基础训练推理测试方法如下所示。


```bash
bash test_tipc/prepare.sh test_tipc/configs/can/train_infer_python.txt lite_train_lite_infer
```

```bash
bash test_tipc/test_train_inference_python.sh test_tipc/configs/can/train_infer_python.txt lite_train_lite_infer
```

输出结果如下，表示命令运行成功。

```bash
Run successfully with command - python tools/train.py --config config_for_tipc.yaml!
......
Run successfully with command - python tools/infer.py --use_gpu=True > ./test_tipc/output/python_infer_gpu_usetrt_null_precision_null_batchsize_null.log 2>&1 !
```
该信息可以在运行log中查看，以`can`为例，log位置在`./test_tipc/output/results_python.log`。

如果运行失败，也会在终端中输出运行失败的日志信息以及对应的运行命令。可以基于该命令，分析运行失败的原因。