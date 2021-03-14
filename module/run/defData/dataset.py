import random
import torch

# Train,Valid用のデータクラス
class TrainDataset:
    def __init__(self, features, targets, p: float = 1., transform=None, transformsAug=None):
        self.features = features
        self.targets = targets

        # 1列のデータを画像としての形に整形
        self.features = self.features.reshape(self.features.shape[0], 3, 32, 32).astype("uint8")
        # augmentation
        self.transform = transform
        self.transformsAug = transformsAug
        self.p = p  # transformの使用有無の判断に使用

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        x = self.features[idx, :]

        if self.transform:
            if random.random() < self.p:  # pの値を大きくするほどtransformの使用率が上がる。
                augmented = self.transform(image=x)
                x = augmented['image']
                del augmented

        X = []  # TTA別に画像をリスト化
        if self.transformsAug:
            for transformAug in self.transformsAug:
                augmented = transformAug(image=x)
                aug_tmp = torch.tensor(augmented['image'], dtype=torch.float)
                X.append(aug_tmp)
                del augmented
        else:
            x = torch.tensor(x, dtype=torch.float)
            X.append(x)

        # torch.DataLoaderに入れるための形式
        dct = {
            'x': X,
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


class ValidDataset:
    def __init__(self, features, targets, transforms=None):
        self.features = features
        self.targets = targets

        # 1列のデータを画像としての形に整形
        self.features = self.features.reshape(self.features.shape[0], 3, 32, 32).astype("uint8")
        # augmentation
        self.transforms = transforms

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        X = []  # TTA別に画像をリスト化
        x = self.features[idx, :]

        if self.transforms:
            for transform in self.transforms:
                augmented = transform(image=x)
                aug_tmp = torch.tensor(augmented['image'], dtype=torch.float)
                X.append(aug_tmp)
                del augmented
        else:
            x = torch.tensor(x, dtype=torch.float)
            X.append(x)

        # torch.DataLoaderに入れるための形式
        dct = {
            'x': X,
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


# Test用のデータクラス
class TestDataset:
    def __init__(self, features):
        self.features = features

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        dct = {
            # torch.DataLoaderに入れるための形式
            'x': torch.tensor(self.features[idx, :], dtype=torch.float)
        }
        return dct
