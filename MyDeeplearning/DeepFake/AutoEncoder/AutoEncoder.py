import torch
import torch.nn as nn
import torch.optim as optim


# 定义自动编码器类
class StackedAutoencoder(nn.Module):
    def __init__(self, input_dims, hindden_dims):
        super(StackedAutoencoder, self).__init__()
        self.input_dim = input_dims
        self.hidden_dims = hindden_dims

        # 定义编码器
        self.encoder1 = nn.Sequential(
            nn.Linear(input_dims, hindden_dims[0]),
            nn.ReLU(True),
            nn.Linear(hindden_dims[0], hindden_dims[1]),
            nn.ReLU(True),
            nn.Linear(hindden_dims[1], hindden_dims[2]),
            nn.ReLU(True)
        )

        # 定义解码器
        self.decoder1 = nn.Sequential(
            nn.Linear(hindden_dims[2], hindden_dims[1]),
            nn.ReLU(True),
            nn.Linear(hindden_dims[1], hindden_dims[0]),
            nn.ReLU(True),
            nn.Linear(in_features=hindden_dims[0], out_features=input_dims),
        )

    def forward(self, x):
        x = self.encoder1(x)
        x = self.decoder1(x)
        return x

    def get_encoder_output(self, x):
        return self.encoder1(x)


def train(model, train_loader, num_epochs, learning_rate):
    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练模型
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for step, data in enumerate(train_loader, start=0):
            inputs, targets = data

            inputs, targets = inputs.reshape(inputs.size(0), -1), targets.reshape(inputs.size(0), -1)
            # 展开为（batch,w*h）唯独
            # 梯度清零
            optimizer.zero_grad()

            # 前向计算
            outputs = model(inputs)
            loss = criterion(outputs, inputs)

            # 反向传播
            loss.backward()
            optimizer.step()
            # 打印损失
            running_loss += loss.item()
            # 打印训练过程
            rate = (step + 1) / len(train_loader)
            a = "*" * int(rate * 50)
            b = "." * int((1 - rate) * 50)
            print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")
        print()
        print("Epoch[{}/{}],loss:{:.4f}".format(epoch + 1, num_epochs, loss.item()))


# 定义测试函数
def test(model, test_loader):
    model.eval()
    criterion = nn.MSELoss()

    # 评估模型
    test_loss = 0
    with torch.no_grad():
        for data in test_loader:
            # Get the input data and target data
            inputs, targets = data
            inputs, targets = inputs.reshape(inputs.size(0), -1), targets.reshape(inputs.size(0), -1)

            # Forward pass
            outputs = model(inputs)
            test_loss += criterion(outputs, inputs).item()

    # Print the average test loss
    test_loss /= len(test_loader.dataset)
    print('Average Test Loss: {:.4f}'.format(test_loss))
