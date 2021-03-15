import numpy as np
from sklearn import preprocessing

#自作app
from module import loading, preprocessing, CV_folds
from module import runKFolds, CVEvaluation
from CIFAR10_pycharm.module.conf import setting, configAboutFitting

import warnings
warnings.filterwarnings('ignore')

#HyperParameter
param_space = {'hidden_size1': 512,
               'hidden_size2': 512,
               'dropOutRate1': 0.20393004966355735,
               'dropOutRate2': 0.39170486751620137,
               'leakyReluSlope': 0.01973893854348531,
              }

def Exec(param):
    # Tester(True/False)
    Tester = True

    # Plot(True/False)
    Plotting = True

    # Preprocessing Data
    train, test, targetOH = preprocessing.Exec(param, loading.trainFeature,
                                             loading.testFeature, loading.trainTarget)

    # CV folds
    folds = CV_folds.Exec(train, loading.trainTarget)

    # Config about Fitting
    confFitting = configAboutFitting.outputConfig(train, test, targetOH, folds)

    # Averaging on multiple SEEDS
    oof = np.zeros((len(train), confFitting["num_targets"]))
    predictions = np.zeros((len(test), confFitting["num_targets"]))

    ### RUN ###
    for seed in setting.SEED:
        print('~' * 20, 'SEED', seed, '~' * 20)
        oof_, predictions_ = runKFolds.Exec(Tester, Plotting, setting.NFOLDS, seed, param,
                                            folds, train, test, targetOH, confFitting)
        oof += oof_ / len(setting.SEED)
        predictions += predictions_ / len(setting.SEED)

    # CV 評価
    score = CVEvaluation.Exec(confFitting, oof, loading.trainTarget, targetOH)

    # 課題提出
    # Submit(confFitting, predictions, test)

    return score, oof, predictions

score, oof, predictions = Exec(param_space)
