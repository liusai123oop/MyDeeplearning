import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import MobileNetV2


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load image
    img_path = "../tulip.jpg"
    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)
    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    # read class_indict
    json_path = './class_indices.json'
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)

    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # create model
    model = MobileNetV2(num_classes=5).to(device)
    # load model weights
    model_weight_path = "./MobileNetV2.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=device))
    model.eval()
    output = torch.squeeze(model(img.to(device)))

    # img是输入的图像数据，to(device)
    # 方法将其转移到模型所在的设备上，这个设备可能是CPU或GPU。
    # model是已经训练好的神经网络模型。
    # model(img.to(device))
    # 将输入图像传递给模型，得到模型的原始输出。
    # torch.squeeze()
    # 函数用于去除输出张量中所有维度为1的维度，这样做通常是为了使得输出的格式更加整洁，便于后续处理。
    # predict = torch.softmax(output, dim=0): 这行代码将模型的原始输出（通常是logits）转换为概率分布。
    #
    # torch.softmax()
    # 函数是实现这个转换的函数，它接受原始输出output作为输入。
    # dim = 0
    # 参数指定了在哪个维度上应用softmax函数。在这个例子中，我们假设output的维度至少是2（例如，[batch_size,
    #                                                                                      num_classes]），dim = 0
    # 意味着对每个样本的类别概率进行归一化。
    # predict_cla = torch.argmax(predict).numpy(): 这行代码从概率分布中找到最可能的类别。
    #
    # torch.argmax()
    # 函数返回输入张量中最大值的索引，这里它作用于predict（即softmax函数的输出）。
    # .numpy()
    # 方法将PyTorch张量转换为NumPy数组，这样可以方便地在Python中使用这个类别索引。
    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img.to(device))).cpu()
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()

    print_res = "class: {}   prob: {:.3}".format(class_indict[str(predict_cla)],
                                                 predict[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict)):
        print("class: {:10}   prob: {:.3}".format(class_indict[str(i)],
                                                  predict[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()