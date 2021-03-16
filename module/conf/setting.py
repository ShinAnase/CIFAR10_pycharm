import torch

INPUT = "/home/tidal/ML_Data/CIFAR10/cifar-10-python/cifar-10-batches-py"
OUTPUT = "/home/tidal/ML_Data/CIFAR10/output"

SUBMIT = OUTPUT + "/submittion/"
SAVEMODEL = OUTPUT + "/model/Pytorch/"
SAVEOOF = OUTPUT + "/OOF/Pytorch/"
SAVEPLOT = OUTPUT + "/plot_history/"
SAVEIMG = OUTPUT + "/plot_img/"

#ARCH = EfficientNet.from_pretrained('efficientnet-b1')

# RUN
DEVICE = ('cuda' if torch.cuda.is_available() else 'cpu')
EPOCHS = 40
TRAIN_BATCH_SIZE = 2048
VALID_BATCH_SIZE = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5
NFOLDS = 5 #本番は5
EARLY_STOPPING_STEPS = 50
EARLY_STOP = False
AUGMENT_PRB = 1
SEED = [42]