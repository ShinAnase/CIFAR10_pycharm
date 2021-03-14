from sklearn.model_selection import StratifiedKFold
from CIFAR10_pycharm.module.conf import setting


def Exec(train, target):
    folds = train.copy()

    skf = StratifiedKFold(n_splits=setting.NFOLDS)

    for f, (t_idx, v_idx) in enumerate(skf.split(X=train, y=target)):
        folds.loc[v_idx, 'kfold'] = int(f)

    folds['kfold'] = folds['kfold'].astype(int)

    return folds