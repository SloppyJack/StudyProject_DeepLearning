# 文件结构
- data: 存在MINIST图片数据
- deeplearning_test
  - cnn_digit_recog: 基于 PyTorch 从头搭建了一个两层卷积神经网络，完成了 MNIST 数据处理、模型训练、训练/验证/测试集划分、最佳模型选择以及卷积核和中间特征图可视化，并分析了 CNN 在训练过程中卷积特征的变化
- test_from_cs231n
  - vanilla_rnn：经典RNN递归网络模型，求解输入两个连续的1