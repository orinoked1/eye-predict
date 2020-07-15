from modules.data.datasets import DatasetBuilder
#from modules.models.torch_nn import TorchNn
from modules.models.binary_NN import BinaryNN
from modules.models.linear_regression_torch import linearRegressionModel
from modules.models.binary_simple_cnn import BinarySimpleCnn
import datetime


datasetbuilder = DatasetBuilder()
#stims_array, scanpath_df, fixation_df = datasetbuilder.processed_data_loader()
#train, val, test = datasetbuilder.train_test_val_split(stimType, scanpath_df, fixation_df, seed)
#train.reset_index(inplace=True)
#val.reset_index(inplace=True)
#test.reset_index(inplace=True)
#print("Log... Train shape", train.shape)
#print("Log... Val shape", val.shape)
#print("Log... Test shape", test.shape)

"""
trainImg, trainX, trainY, stim_size = datasetbuilder.preper_data_for_model(train, stimType, scanpath_lan, is_scanpath=True,
                                                                    is_fixation=False,
                                                                    is_coloredpath=False, color_split=110, is_img=False)
valImg, valX, valY, stim_size  = datasetbuilder.preper_data_for_model(val, stimType, scanpath_lan, is_scanpath=True,
                                                                 is_fixation=False,
                                                                 is_coloredpath=False, color_split=110, is_img=False)
testImg, testX, testY, stim_size = datasetbuilder.preper_data_for_model(test, stimType, scanpath_lan, is_scanpath=True,
                                                                  is_fixation=False,
                                                                  is_coloredpath=False, color_split=110, is_img=False)
"""
def run_binary_nn_model(seed, stimType, bin_count, stims_array, scanpath_df, fixation_df):

    scanpath_lan = 3000
    color_split = None
    is_scanpath = True
    is_fixation = False
    is_coloredpath = False
    is_img = False


    trainImg, trainX, trainY, valImg, valX, valY, stim_size = \
        datasetbuilder.get_train_dev_data_for_model_run(stims_array, scanpath_df, fixation_df, stimType, seed, scanpath_lan, color_split,
                                                                                       is_scanpath,
                                                                                       is_fixation,
                                                                                       is_coloredpath,
                                                                                       is_img
                                                        , bin_count)

    run_name = datetime.datetime.now()
    binary_nn = BinaryNN(seed, run_name, stim_size, scanpath_lan, stimType, bin_count)
    # Build train and evaluate model
    binary_nn.define_model()
    binary_nn.train_model(trainX, trainY, valX, valY)
    binary_nn.test_model(valX, valY)
    binary_nn.metrices()

    print('zzz')

    return


def run_simple_cnn_model(seed, stimType):
    scanpath_lan = None
    color_split = 110
    is_scanpath = True
    is_fixation = False
    is_coloredpath = False
    is_img = False

    trainImg, trainX, trainY, valImg, valX, valY, stim_size = \
        datasetbuilder.get_train_dev_data_for_model_run(stimType, seed, scanpath_lan, color_split,
                                                        is_scanpath,
                                                        is_fixation,
                                                        is_coloredpath,
                                                        is_img)

    run_name = datetime.datetime.now()
    binary_simple_cnn = BinarySimpleCnn(seed, run_name, stim_size, color_split)
    # Build train and evaluate model
    binary_simple_cnn.define_model()
    binary_simple_cnn.train_model(trainX, trainY, valX, valY)
    binary_simple_cnn.test_model(valX, valY)
    binary_simple_cnn.metrices()

    return

def run_linearRegTorch_model(seed, stimType, bin_count, stims_array, scanpath_df, fixation_df):

    scanpath_lan = 1000
    color_split = None
    is_scanpath = True
    is_fixation = False
    is_coloredpath = False
    is_img = False


    trainImg, trainX, trainY, valImg, valX, valY, stim_size = \
        datasetbuilder.get_train_dev_data_for_model_run(stims_array, scanpath_df, fixation_df, stimType, seed, scanpath_lan, color_split,
                                                                                       is_scanpath,
                                                                                       is_fixation,
                                                                                       is_coloredpath,
                                                                                       is_img
                                                        , bin_count)

    run_name = datetime.datetime.now()
    linearReg = linearRegressionModel(seed, run_name, stim_size, scanpath_lan, stimType, bin_count)
    # Build train and evaluate model
    linearReg.define_model(trainX)
    linearReg.train_model(trainX, trainY, valX, valY)
    linearReg.test_model(trainX, trainY)
    linearReg.metrices(trainY)
    return


#2 bin -> 10 bin -> 5 bin -> 10 bin

#seed = 1010
stims_array, scanpath_df, fixation_df =  datasetbuilder.load_data_pickle()
stimType = "Face"

#for bin_count in [2, 5, 10]:
bin_count = 5
for seed in range(20130, 20131):
    #run_binary_nn_model(seed, stimType, bin_count, stims_array, scanpath_df, fixation_df)
    run_linearRegTorch_model(seed, stimType, bin_count, stims_array, scanpath_df, fixation_df)