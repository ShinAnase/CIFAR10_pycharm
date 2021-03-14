import numpy as np
from CIFAR10_pycharm.module.run import singleFoldRunning


def Exec(Tester, Plotting, NFOLDS, seed, param,
               folds, train, test, target, confFitting):
    oof = np.zeros((len(train), confFitting["num_targets"]))
    predictions = np.zeros((len(test), confFitting["num_targets"]))

    for fold in range(NFOLDS):
        print('=' * 20, 'Fold', fold, '=' * 20)
        oof_, pred_ = singleFoldRunning.run_training(confFitting, Tester, Plotting,
                                                     fold, seed, param, folds,
                                                     train, test, target)

        predictions += pred_ / NFOLDS
        oof += oof_

    return oof, predictions