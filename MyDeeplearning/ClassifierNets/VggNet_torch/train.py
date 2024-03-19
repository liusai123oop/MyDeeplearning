import json
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
from torchvision import transforms, datasets, utils
import torch.optim as optim
from tqdm import tqdm

from model import vgg
import os
import torch
import torch.nn as nn

# 训练参数配置
data_filename = "flower_data"
batch_size = 32
learning_rate = 0.0002
epochs = 10
num_classes = 5
init_weights=True
model_name = "vgg16"
device = torch.device("cuda:o" if torch.cuda.is_available() else "cpu")
print(device)

# 设置数据预处理操作
data_transform = {
    "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                 transforms.RandomHorizontalFlip(),
                                 transforms.ToTensor(),
                                 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
    "val": transforms.Compose([transforms.Resize((224, 224)),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
}
# 确定当前工作目录（os.getcwd()）。
# 从当前工作目录向上移动两级目录（"../.."）。
# 获取这个新路径的绝对路径（os.path.abspath）
# 配置数据集的位置
data_root = os.path.abspath(os.path.join(os.getcwd(), "../.."))
image_path = os.path.join(data_root, "data_set")
image_path = os.path.join(image_path, data_filename)
root = os.path.join(image_path, "train")
train_dataset = datasets.ImageFolder(root=os.path.join(image_path, "train"), transform=data_transform["train"])
train_num = len(train_dataset)

classes_list = train_dataset.class_to_idx
print("标签映射->", classes_list)
# key和val互换位置
classes_dict = dict((val, key) for key, val in classes_list.items())

# 写进json文件中，将类别映射关系,indent=4,缩进四个空格
json_str = json.dumps(classes_dict, indent=4)
with open("classes_indices.json", 'w') as json_file:
    json_file.write(json_str)

# 加载数据读取器
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size=batch_size, shuffle=True,
                                           num_workers=0)
validate_dataset = datasets.ImageFolder(root=os.path.join(image_path, "val"),
                                        transform=data_transform["val"])
val_num = len(validate_dataset)
validate_loader = torch.utils.data.DataLoader(validate_dataset,
                                              batch_size=4, shuffle=True,
                                              num_workers=0)

# 绘制几个数据
# test_data_iter = iter(validate_loader)
# test_image,test_labels = test_data_iter.__next__()
#
# def imshow(img):
#     img = img/2+0.5 #unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg,(1,2,0)))
#     plt.show()
#
# print(' '.join('%5s' % classes_dict[test_labels[j].item()] for j in range(4)))
# imshow(utils.make_grid(test_image))
#
net = vgg(model_name=model_name,num_classes=num_classes, init_weights=init_weights)

net.to(device)

# 损失函数设置
loss_function = nn.CrossEntropyLoss()

# 优化器设置
optimizer = optim.Adam(net.parameters(), lr=learning_rate)

# 模型保存路径
folder_path = "./logs"
if not os.path.exists(folder_path):
    # 如果不存在，创建文件夹
    os.makedirs(folder_path)
    print(f"Folder '{folder_path}' created.")
else:
    print(f"Folder '{folder_path}' already exists.")
save_path = 'logs/bestmodel.pth'
best_acc = 0.0
train_steps= len(train_loader)
for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    t1 = time.perf_counter()  # 开始计时
    # train_bar = tqdm(train_loader, file=sys.stdout)
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        images, labels = data
        # 清楚之前的的梯度
        optimizer.zero_grad()
        # 前向计算
        outputs = net(images.to(device))
        loss = loss_function(outputs, labels.to(device))
        # 反向传播
        loss.backward()
        optimizer.step()
        # 打印损失
        running_loss += loss.item()
        # 打印训练过程
        # rate = (step + 1) / len(train_loader)
        # a = "*" * int(rate * 50)
        # b = "." * int((1 - rate) * 50)
        # print("\rtrain loss: {:^3.0f}%[{}->{}]{:.3f}".format(int(rate * 100), a, b, loss), end="")

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                 epochs,
                                                                 loss)
    print()
    print(time.perf_counter() - t1)

    # 验证集
    net.eval()
    acc = 0.0
    with torch.no_grad():
        # 调用新的接口处理打印过程
        val_bar = tqdm(validate_loader, file=sys.stdout)
        for val_data in val_bar:
            val_images, val_labels = val_data
            outputs = net(val_images.to(device))
            predict_y = torch.max(outputs, dim=1)[1]
            # acc += (predict_y == test_labels.to(device)).sum().item()
            acc += torch.eq(predict_y, val_labels.to(device)).sum().item()
            val_accurate = acc / val_num
            if val_accurate > best_acc:
                best_acc = val_accurate
                torch.save(net.state_dict(), save_path)
        # print('[epoch %d] train_loss: %.3f test_accuracy: %.3f' % (epoch + 1, running_loss / step, acc / val_num))
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))
print("Finished Training")
