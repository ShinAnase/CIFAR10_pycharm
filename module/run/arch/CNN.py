import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, num_features, num_targets, param):
        super(Model, self).__init__()
        self.hidden_size1 = param['hidden_size1']
        dropOutRate1 = param['dropOutRate1']
        # -------------------前準備---------------------------------------------------------------------
        self.layersCompo = lambda in_ch, out_ch: [torch.nn.Conv2d(in_ch,  # チャネル入力（色の部分）
                                                                  out_ch,  # チャンネル出力
                                                                  3,  # カーネルサイズ(フィルタサイズ)
                                                                  1,  # ストライド (デフォルトは1)
                                                                  1,  # パディング (デフォルトは0)
                                                                  ),
                                                  nn.BatchNorm2d(out_ch),
                                                  nn.ReLU(),
                                                  nn.Dropout(p=dropOutRate1)]

        layersList = []

        layersList.extend(self.layersCompo(3, 64))
        layersList.extend(self.layersCompo(64, 64))
        layersList.extend(self.layersCompo(64, 64))
        layersList.append(torch.nn.AvgPool2d(2))  # カーネルサイズ

        layersList.extend(self.layersCompo(64, 128))
        layersList.extend(self.layersCompo(128, 128))
        layersList.extend(self.layersCompo(128, 128))
        layersList.append(torch.nn.AvgPool2d(2))  # カーネルサイズ

        layersList.extend(self.layersCompo(128, self.hidden_size1))
        layersList.extend(self.layersCompo(self.hidden_size1, self.hidden_size1))
        layersList.extend(self.layersCompo(self.hidden_size1, self.hidden_size1))

        # ------------------モデル--------------------------------------------------------------------------
        self.layers = nn.ModuleList(layersList)
        # F.avg_pool2d(x, kernel_size=x.size()[2:]) #Grobal average pooling　（画像面ごとにまとめる）

        self.dense1 = nn.Linear(self.hidden_size1, num_targets)

    def forward(self, x):
        for i in range(len(self.layers)):
            # print(x.shape)
            x = self.layers[i](x)
        x = F.avg_pool2d(x, kernel_size=x.size()[2:])

        x = x.view(-1, self.hidden_size1)  # [512,256,1,1]->[512,256]
        # print(x.shape)
        x = self.dense1(x)

        return x