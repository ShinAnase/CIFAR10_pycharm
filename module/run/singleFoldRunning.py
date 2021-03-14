import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

#自作app
from CIFAR10_pycharm.module.run.defData import dataset as Dt, AugUtl as Aug
from CIFAR10_pycharm.module.run.arch import CNN
from CIFAR10_pycharm.module.run.runUtl import runFn
from CIFAR10_pycharm.module.conf import setting


# Seed固定
def seed_everything(seed=42):
    # data取得についてのランダム性固定
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # cudnnによる演算の安定化(評価値の安定)
    torch.backends.cudnn.deterministic = True

    # os.environ['PYTHONHASHSEED'] = str(seed)

def run_training(confFitting, Tester, Plotting, fold, seed, param,
                 folds, train, test, target):

    seed_everything(seed)

    train = folds

    trn_idx = train[train['kfold'] != fold].index
    val_idx = train[train['kfold'] == fold].index

    train_df = train[train['kfold'] != fold].reset_index(drop=True)
    valid_df = train[train['kfold'] == fold].reset_index(drop=True)

    x_train, y_train  = train_df[confFitting["feature_cols"]].values, train_df[confFitting["target_cols"]].values
    x_valid, y_valid =  valid_df[confFitting["feature_cols"]].values, valid_df[confFitting["target_cols"]].values

    # aumentation
    transforms = Aug.get_transform()
    train_augmentation = Aug.get_augmentation_Train()
    valid_augmentation = Aug.get_augmentation_TTA()

    train_dataset = Dt.TrainDataset(x_train, y_train, setting.AUGMENT_PRB, transforms, train_augmentation)
    valid_dataset = Dt.TrainDataset(x_valid, y_valid, setting.AUGMENT_PRB, transforms, valid_augmentation)
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=setting.TRAIN_BATCH_SIZE, shuffle=True)
    validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=setting.VALID_BATCH_SIZE, shuffle=False)

    model = CNN.Model(
        num_features=confFitting["num_features"],
        num_targets=confFitting["num_targets"],
        # arch=ARCH,
        param=param
    )

    model.to(setting.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=setting.LEARNING_RATE, weight_decay=setting.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, pct_start=0.1, div_factor=1e3,
                                              max_lr=1e-2, epochs=setting.EPOCHS, steps_per_epoch=len(trainloader) * len(train_augmentation))
    # scheduler = sch.CosineAnnealingWarmupRestarts(optimizer,
    #                                              first_cycle_steps=200,
    #                                              cycle_mult=1.0,
    #                                              max_lr=0.01,
    #                                              min_lr=0.0001,
    #                                              warmup_steps=50,
    #                                              gamma=0.5)

    ##### 評価関数 ######
    train_loss_fn = nn.BCEWithLogitsLoss()
    valid_loss_fn = nn.BCEWithLogitsLoss()

    early_stopping_steps = setting.EARLY_STOPPING_STEPS
    early_step = 0

    oof = np.zeros((len(train), target.shape[1]))
    best_loss = np.inf
    history = {
        'train_loss': [],
        'valid_loss': [],
    }
    fn = runFn.runFunc()
    best_epoch = 0.
    best_train_loss = 0.

    for epoch in range(setting.EPOCHS):

        train_loss = fn.train_fn(model, optimizer, scheduler, train_loss_fn, trainloader, setting.DEVICE)
        valid_loss, valid_preds = fn.valid_fn(model, valid_loss_fn, validloader, setting.DEVICE)
        if Tester:
            print("EPOCH: {:03}: | train_loss: {:.3f}: | valid_loss: {:.3f}".format(epoch, train_loss, valid_loss))


        history['train_loss'].append(train_loss)
        history['valid_loss'].append(valid_loss)

        if valid_loss < best_loss:
            best_epoch = epoch
            best_train_loss = train_loss
            best_loss = valid_loss
            oof[val_idx] = valid_preds
            torch.save(model.state_dict(), f"{setting.SAVEMODEL}SEED{seed}_FOLD{fold}.pth")

        elif(setting.EARLY_STOP == True):
            early_step += 1
            if (early_step >= early_stopping_steps):
                if Tester:
                    print('Early stopping. Best Val loss: {:.3f}'.format(best_loss))
                break

    print("<BEST LOSS> EPOCH: {:03}: | train_loss: {:.3f}: | valid_loss: {:.3f}".format(best_epoch, best_train_loss, best_loss))

    # Visuarization
    if Plotting:
        plt.plot(range(1, len(history['train_loss']) + 1), history['train_loss'], label='train_loss')
        plt.plot(range(1, len(history['valid_loss']) + 1), history['valid_loss'], label='valid_loss')
        plt.title(f'Seed{seed} Fold{fold} LOSS VISUARIZATION')
        plt.xlabel('epoch')
        plt.ylabel('loss')
        plt.legend()
        plt.savefig(f'{setting.SAVEPLOT}Seed{seed}_Fold{fold}_history.png')
        plt.close()

    del history

    # --------------------- PREDICTION---------------------
    # x_test = test[confFitting["feature_cols"]].values
    # testdataset = TestDataset(x_test)
    # testloader = torch.utils.data.DataLoader(testdataset, batch_size=BATCH_SIZE, shuffle=False)
    #
    # model = Model(
    #    num_features=confFitting["num_features"],
    #    num_targets=confFitting["num_targets"],
    #    arch=ARCH,
    #    param=param
    # )
    #
    # model.load_state_dict(torch.load(f"{SAVEMODEL}SEED{seed}_FOLD{fold}.pth"))
    # model.to(DEVICE)
    #
    predictions = np.zeros((len(test), confFitting["num_targets"]))
    # predictions = inference_fn(model, testloader, DEVICE)


    return oof, predictions
