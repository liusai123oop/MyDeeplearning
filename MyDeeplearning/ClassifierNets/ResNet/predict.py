import os
import json

import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import resnet34


def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    data_transform = transforms.Compose(
        [transforms.Resize(256),
         transforms.CenterCrop(224),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # 加载图片
    img_path = ""
    assert os.path.exists(img_path), "file:'{}' does not exists.".format(img_path)
    img = Image.open(img_path)
    plt.imshow(img)

    # 预处理图片
    img = data_transform(img)

    # 添加batch维度
    img = torch.unsqueeze(img, dim=0)

    # 读取标签
    json_path = ""

    with open(json_path, "r") as f:
        classes_indict = json.load(f)

    # 加载模型
    model = resnet34(num_classes=5).to(device)

    weights_path=""
    assert os.path.exists(weights_path), "file:'{}' does not exists".format(weights_path)
    model.load_state_dict(torch.load(weights_path, map_location=device))

    model.eval()
    with torch.no_grad():
        outputs = torch.squeeze(model(img.to(device))).cpu
        predict_y = torch.softmax(outputs,dim=0)
        predict_cla = torch.argmax(predict_y).numpy()

    print_res = "class:{} prob: {:.3}".format(classes_indict[predict_cla],predict_y[predict_cla].numpy())
    plt.title(print_res)
    for i in range(len(predict_y)):
        print("class: {:10}   prob: {:.3}".format(classes_indict[str(i)],
                                                  predict_y[i].numpy()))
    plt.show()


if __name__ == '__main__':
    main()