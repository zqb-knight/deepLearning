# 深度学习要点

## 一. 主体框架

### 1. Prepare dataset 

准备数据集

### 2. Design model using Class

设计一个模型类，这个模型类要继承于`torch.nn.Module`

注意：

- 模型类需要自己实现forward函数

### 3. Construct loss and optimizer

构建损失和优化器，这里使用`Pytorch API`

### 4. Training cycle

构造训练循环，主要步骤包括：

1. forward 前馈

   利用**forward**函数计算预测值

2. 计算loss

   通过预测值和真实值计算**loss**

3. backward 反馈

   利用**反向传播算法**进行梯度计算

4. update 更新参数

   利用**梯度下降算法**更新参数