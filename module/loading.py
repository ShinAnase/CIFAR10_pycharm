import pickle
import numpy as np
import pandas as pd

from CIFAR10_pycharm.module.conf import setting


#Loading
def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

metaData = unpickle(setting.INPUT + '/batches.meta')
batch1 = unpickle(setting.INPUT + '/data_batch_1')
batch2 = unpickle(setting.INPUT + '/data_batch_2')
batch3 = unpickle(setting.INPUT + '/data_batch_3')
batch4 = unpickle(setting.INPUT + '/data_batch_4')
batch5 = unpickle(setting.INPUT + '/data_batch_5')
testBatch = unpickle(setting.INPUT + '/test_batch')

#trainは分割されていたのを一つに結合(この時点ではまだseries)
trainFeature = np.concatenate([batch1[b'data'],
                               batch2[b'data'],
                               batch3[b'data'],
                               batch4[b'data'],
                               batch5[b'data'],])
testFeature = testBatch[b'data']

trainTarget = np.concatenate([batch1[b'labels'],
                              batch2[b'labels'],
                              batch3[b'labels'],
                              batch4[b'labels'],
                              batch5[b'labels'],])
testTarget = np.array(testBatch[b'labels'])


#pandasとして扱う。
trainFeature = pd.DataFrame(trainFeature)
trainTarget = pd.Series(trainTarget)
testFeature = pd.DataFrame(testFeature)
testTarget = pd.Series(testTarget)
