import os
import torch
import json
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt

from model import GoogLeNet

num_classes = 5
aux_logits = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
weights_path = ""
img_path = ""


def main():
    data_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ]
    )

    # load image

    assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
    img = Image.open(img_path)

    plt.imshow(img)

    img = data_transform(img)

    # 添加batch维度
    img = torch.unsqueeze(img, dim=0)

    # 读取标签
    json_path = "./class_indices.json"
    assert os.path.exists(json_path), "file: '{}' dose not exist.".format(json_path)
    with open(json_path, "r") as f:
        class_indict = json.load(f)

    # 建立模型
    model = GoogLeNet(num_classes=num_classes, aux_logits=aux_logits).to(device)

    # 导入模型参数

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
    missing_keys, unexpected_keys = model.load_state_dict(torch.load(weights_path, map_location=device),
                                                          strict=False)
# strict=False表示允许加载部分参数，如果预训练模型的参数字典中有一些键在当前模型中不存在，则会将这些缺失的键忽略掉。
    model.eval()
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
