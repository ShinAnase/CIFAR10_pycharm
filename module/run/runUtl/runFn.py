import torch
import matplotlib.pyplot as plt
import numpy as np

class runFunc:
    def __init__(self, savePath):
        self.IMG_VSL_FLG_TRAIN = True
        self.IMG_VSL_FLG_VALID = True
        self.savePath = savePath

    def train_fn(self, model, optimizer, scheduler, loss_fn, dataloader, device):
        model.train()
        final_loss = 0

        for data in dataloader:
            targets = data['y'].to(device)
            outputs = torch.zeros_like(targets)
            augLen: int = len(data['x'])

            # Augmentation Imageごとに予測
            for i, dataAug in enumerate(data['x']):
                # imageの可視化(最初のデータだけ)
                if self.IMG_VSL_FLG_TRAIN:
                    img = dataAug[0].detach().cpu().numpy().transpose(1, 2, 0).astype("uint8").copy()
                    plt.axis('off')
                    plt.imshow(img)
                    plt.savefig(f'{self.savePath}Train_img_{i}.png')
                    plt.close()
                    del img

                optimizer.zero_grad()
                inputs = dataAug.to(device)
                outputs = model(inputs)
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()
                scheduler.step()
                final_loss += loss.item()

            self.IMG_VSL_FLG_TRAIN = False

        final_loss = final_loss / (len(dataloader) * augLen)

        return final_loss

    def valid_fn(self, model, loss_fn, dataloader, device):
        model.eval()
        final_loss = 0
        valid_preds = []

        # batchごとの処理
        for data in dataloader:
            targets = data['y'].to(device)
            outputs = torch.zeros_like(targets)

            # Augmentation Imageごとに予測
            for i, dataAug in enumerate(data['x']):
                # imageの可視化(最初のデータだけ)
                if self.IMG_VSL_FLG_VALID:
                    img = dataAug[0].detach().cpu().numpy().transpose(1, 2, 0).astype("uint8").copy()
                    plt.axis('off')
                    plt.imshow(img)
                    plt.savefig(f'{self.savePath}Valid_img_{i}.png')
                    plt.close()
                    del img

                inputs = dataAug.to(device)
                output = model(inputs)
                outputs += output

            self.IMG_VSL_FLG_VALID = False
            outputs /= len(data['x'])  # それぞれの予測を平均化（Test Time Augmentation）
            loss = loss_fn(outputs, targets)

            final_loss += loss.item()
            valid_preds.append(outputs.sigmoid().detach().cpu().numpy())

        final_loss /= len(dataloader)
        valid_preds = np.concatenate(valid_preds)

        return final_loss, valid_preds

    def inference_fn(self, model, dataloader, device):
        model.eval()
        preds = []

        for data in dataloader:
            inputs = data['x'].to(device)

            with torch.no_grad():
                outputs = model(inputs)

            preds.append(outputs.sigmoid().detach().cpu().numpy())

        preds = np.concatenate(preds)

        return preds

