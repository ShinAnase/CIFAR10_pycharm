import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import os
import copy
import seaborn as sns

from sklearn import preprocessing
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision

#自作app
import loading
import preprocessing



import warnings
warnings.filterwarnings('ignore')

import tidalUtl.PrpUtl as prp
import tidalUtl.EdaUtl as eda
import tidalUtl.Scheduler as sch

import albumentations as albu


def Exec(param):
    # Tester(True/False)
    Tester = True

    # Plot(True/False)
    Plotting = True

    # Preprocessing Data
    train, test, target = preprocessing.Exec(param, loading.trainFeature,
                                        loading.testFeature, loading.trainTarget)

    # CV folds
    folds = CV_folds(train, target)

    # Config about Fitting
    confFitting = Config_about_Fitting(train, test, target, folds)

    # Averaging on multiple SEEDS
    SEED = [42]
    oof = np.zeros((len(train), confFitting["num_targets"]))
    predictions = np.zeros((len(test), confFitting["num_targets"]))

    ### RUN ###
    for seed in SEED:
        print('~' * 20, 'SEED', seed, '~' * 20)
        oof_, predictions_ = run_k_fold(Tester, Plotting, NFOLDS, seed, param,
                                        folds, train, test, target, confFitting)
        oof += oof_ / len(SEED)
        predictions += predictions_ / len(SEED)

    # CV 評価
    score = CV_Evaluation(confFitting, oof, target)

    # 課題提出
    # Submit(confFitting, predictions, test)

    return score, oof, predictions

score, oof, predictions = Exec(param_space)
#