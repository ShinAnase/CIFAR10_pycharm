import numpy as np
from sklearn import preprocessing
import optuna
import time

#自作app
from module import loading, preprocessing, CV_folds
from module import runKFolds, CVEvaluation
from CIFAR10_pycharm.module.conf import setting, configAboutFitting

import warnings
warnings.filterwarnings('ignore')


def Exec(trial):
    #Hyper parameter
    param = {'hidden_size1': trial.suggest_categorical('hidden_size1', [128, 256, 512]),
             'dropOutRate1': trial.suggest_uniform('dropOutRate1', 0.01, 0.5),
             }
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

    return score["Accuracy"]
    #return score, oof, predictions

start = time.time()#時間計測用タイマー開始
#score = Exec(optuna.trial.FixedTrial({'hidden_size1': 256,
#                                      'dropOutRate1': 0.3}))
study = optuna.create_study()
study.optimize(Exec, n_trials=7)
print(f"best param: {study.best_params}")
print(f"best score: {study.best_value}")

elapsed_time = (time.time() - start)/60/60
print(f"Time：{elapsed_time}[h]")
