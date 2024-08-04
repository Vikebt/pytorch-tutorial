import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
import matplotlib.pyplot as plt

# 定义神经网络，这里使用简单的全连接网络，输入为28*28的图像，输出为10个类别，
class Net(torch.nn.Module):

    def __init__(self):
        super().__init__()
        # 输入为28*28的图像，输出为10个类别，四层全连接网络。
        self.fc1 = torch.nn.Linear(28*28, 64)
        self.fc2 = torch.nn.Linear(64, 64)
        self.fc3 = torch.nn.Linear(64, 64)
        self.fc4 = torch.nn.Linear(64, 10)
    
    # 定义前向传播过程，输入为x，输出为预测结果
    # 输入为15*28*28的tensor，经过全连接层后输出为15*64的tensor
    # 再经过全连接层后输出为15*64的tensor，再经过全连接层后输出为15*10的tensor
    # 最后使用log_softmax函数归一化操作计算预测结果
    def forward(self, x):
        x = torch.nn.functional.relu(self.fc1(x))
        x = torch.nn.functional.relu(self.fc2(x))
        x = torch.nn.functional.relu(self.fc3(x))
        x = torch.nn.functional.log_softmax(self.fc4(x), dim=1)
        return x

# 获取数据集并进行下载，这里使用MNIST数据集，训练集和测试集各50000个样本，
# 输入为训练集和测试集，返回数据加载器，每次读取15个样本batch_size，
# 打乱数据shuffle，每次训练前将梯度归零。
def get_data_loader(is_train):
    to_tensor = transforms.Compose([transforms.ToTensor()])
    data_set = MNIST("", is_train, transform=to_tensor, download=True)
    return DataLoader(data_set, batch_size=15, shuffle=True)

# 评估模型，输入为测试数据集和神经网络，返回准确率，
# 每次读取15个样本，计算预测结果和真实标签的匹配数量，除以总数量得到准确率。
def evaluate(test_data, net):
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for (x, y) in test_data:
            outputs = net.forward(x.view(-1, 28*28))
            for i, output in enumerate(outputs):
                if torch.argmax(output) == y[i]:
                    n_correct += 1
                n_total += 1
    return n_correct / n_total

# 训练模型，输入为训练数据集和神经网络，
# 每次读取15个样本，计算损失函数值，反向传播梯度，更新参数，
# 打印训练过程，最后评估测试集准确率。
def main():
    # 导入训练集和测试集数据，初始化神经网络
    train_data = get_data_loader(is_train=True)
    test_data = get_data_loader(is_train=False)
    net = Net()
    # 使用交叉熵损失函数，打印初始准确率，测试集准确率在0.1左右
    print("initial accuracy:", evaluate(test_data, net))
    # 定义优化器，使用Adam优化器，学习率为0.001
    optimizer = torch.optim.Adam(net.parameters(), lr=0.001)
    # 训练10个epoch，每轮训练15个batch，每次batch大小为15
    # 每次训练后评估测试集准确率，打印训练过程
    # 最后打印测试集准确率，测试集准确率在0.98左右，说明模型训练成功，可以用于预测
    for epoch in range(10):
        for (x, y) in train_data:
            net.zero_grad()  # 每次训练前将梯度归零，防止梯度累加
            output = net.forward(x.view(-1, 28*28))  # 正向传播，输入为15*28*28的tensor
            loss = torch.nn.functional.nll_loss(output, y)  # 计算损失函数值
            loss.backward()  # 反向误差传播，计算梯度
            optimizer.step()  # 优化网络参数
        print("epoch", epoch, "accuracy:", evaluate(test_data, net))
    # 绘制预测结果，每次读取15个样本，下面使用matplotlib库绘制预测结果，预测结果为10个数字的图像，绘制前三个预测结果，
    # 预测结果为10个数字的图像，预测正确率在95%左右，说明模型预测准确率较高，
    # 也可以继续优化网络结构，增加隐藏层，增加神经元数量等，也可以尝试使用卷积神经网络等。
    for (n, (x, _)) in enumerate(test_data):
        if n > 3:
            break
        predict = torch.argmax(net.forward(x[0].view(-1, 28*28)))
        plt.figure(n)
        plt.imshow(x[0].view(28, 28))
        plt.title("prediction: " + str(int(predict)))
    plt.show()


if __name__ == "__main__":
    main()
