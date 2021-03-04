# 利用糖尿病数据实现二分类
# 练习使用多维数据的导入，练习多层神经网络的构建
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)
x_data = torch.from_numpy(xy[:, :-1])
y_data = torch.from_numpy(xy[:, [-1]])


# 准备数据集
class DiabetsDataset(Dataset):
    def __init__(self):
        self.xy = np.loadtxt("diabetes.csv.gz", delimiter=",", dtype=np.float32)
        self.len = xy.shape[0]
        self.x_data = torch.from_numpy(xy[:, :-1])
        self.y_data = torch.from_numpy(xy[:, [-1]])

    def __getitem__(self, item):
        return self.x_data[item], self.y_data[item]

    def __len__(self):
        return self.len


# 构建模型
class MutiModel(torch.nn.Module):
    def __init__(self):
        super(MutiModel, self).__init__()
        self.Linear_8to6 = torch.nn.Linear(8, 6)
        self.Linear_6to4 = torch.nn.Linear(6, 4)
        self.Linear_4to1 = torch.nn.Linear(4, 1)
        self.activate = torch.nn.Sigmoid()

    def forward(self, x):
        x = self.activate(self.Linear_8to6(x))
        x = self.activate(self.Linear_6to4(x))
        x = self.activate(self.Linear_4to1(x))
        return x


dataset = DiabetsDataset()
train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          num_workers=2,
                          shuffle=True)

model = MutiModel()
criterion = torch.nn.BCELoss(size_average=True)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

if "__name__" == "__main__":
    for epoch in range(1000):
        for index, data in enumerate(train_loader, 0):
            # forward
            input, labels = data
            y_pred = model(input)
            loss = criterion(y_pred, labels)
            print(epoch, loss)
            # backward
            optimizer.zero_grad()
            loss.backward()
            # update
            optimizer.step()

x_test = torch.Tensor([[-0.882353, -0.145729, 0.0819672, -0.414141, 0, -0.207153, -0.766866, -0.666667]])
y_test = model(x_test)
print("After training:", y_test.data)
