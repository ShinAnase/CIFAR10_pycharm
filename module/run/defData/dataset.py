import random
import torch
import albumentations as albu

# Train,Valid用のデータクラス
class TrainDataset:
    def __init__(self, features, targets, p: float = 1.):
        self.features = features
        self.targets = targets

        # 1列のデータを画像としての形に整形
        self.features = self.features.reshape(self.features.shape[0], 3, 32, 32).astype("uint8")
        # augmentation
        self.p = p  # transformの使用有無の判断に使用

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        x = self.features[idx, :]


        if random.random() < self.p:  # pの値を大きくするほどtransformの使用率が上がる。
            x = albu.HorizontalFlip(p=1)(image=x)['image']

        # torch.DataLoaderに入れるための形式
        dct = {
            'x': torch.tensor(x, dtype=torch.float),
            'y': torch.tensor(self.targets[idx, :], dtype=torch.float)
        }
        return dct


class ValidDataset:
    def __init__(self, features, targets, p: float = 1.):
        self.features = features
        self.targets = targets

        # 1列のデータを画像としての形に整形
        self.features = self.features.reshape(self.features.shape[0], 3, 32, 32).astype("uint8")
        # augmentation
        self.p = p  # transformの使用有無の判断に使用

    def __len__(self):
        return (self.features.shape[0])

    def __getitem__(self, idx):
        x = self.features[idx, :]

        transformsAugs = [
            #[albu.ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)],
            [],
        ]

        X = []  # TTA別に画像をリスト化
        for transformAug in transformsAugs:
            augmented = albu.Compose(transformAug)(image=x)
            aug_tmp = torch.tensor(augmented['image'], dtype=torch.float)
            X.append(aug_tmp)
            del augmented

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
