import sys
sys.path.append('../')
import pandas as pd
from modules.data.preprocessing import DataPreprocess
from modules.data.datasets import DatasetBuilder
from modules.data.stim import Stim
from modules.models.cnn_lstm import CnnLstm
from modules.models.simple_lstm import SimpleLstm
from modules.models.svm import SVM
from modules.models.cnn_lstm_img_concat import CnnLstmImgConcat
from modules.models.cnn_multi_input import CnnMultiInput
from modules.models.cnn_multi_input_vggnet import CnnMultiInputVGGNet
import os
import yaml

def get_datasets():
    path = os.getcwd()
    expconfig = "/modules/config/experimentconfig.yaml"
    with open(path + expconfig, 'r') as ymlfile:
        cfg = yaml.load(ymlfile)

    stimSnack = Stim(cfg['exp']['etp']['stimSnack']['name'], cfg['exp']['etp']['stimSnack']['id'], cfg['exp']['etp']['stimSnack']['size'])
    stimFace = Stim(cfg['exp']['etp']['stimFace']['name'], cfg['exp']['etp']['stimFace']['id'], cfg['exp']['etp']['stimFace']['size'])
    stimArray = [stimFace, stimSnack]
    data = DataPreprocess(cfg['exp']['etp']['name'],
                          cfg['exp']['etp']['both_eye_path'],
                          cfg['exp']['etp']['one_eye_path1'],
                          cfg['exp']['etp']['trial_start_str'],
                          cfg['exp']['etp']['trial_end_str'],
                          cfg['exp']['etp']['output_file_both_eye'],
                          cfg['exp']['etp']['output_file_one_eye1'], [stimSnack, stimFace])
    fixation_only = False
    datasetbuilder = DatasetBuilder([stimSnack, stimFace])

    try:
        print("Log... reading fixation and scanpath df's")
        fixation_df = pd.read_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
        scanpath_df = pd.read_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")
    except:
        try:
            print("Log... reading tidy df")
            tidy_data = pd.read_pickle(path + "/etp_data/processed/tidy_data_40_subjects.pkl")
            fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
            scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
            fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
            scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")
        except:
            try:
                print("Log... reading raw data csv")
                both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
                one_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_one_eye_data.csv") #(one_eye_data_path)
                all_data = pd.concat([both_eye_data, one_eye_data])
                tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
                tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_40_subjects.pkl")
                fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
                scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
                fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
                scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")
            except:
                print("Log... processing raw data to csv")
                both_eye_data_path = data.read_eyeTracking_data_both_eye_recorded(fixation_only)
                one_eye_data_path = data.read_eyeTracking_data_one_eye_recorded(fixation_only)
                both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
                one_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_one_eye_data.csv") #(one_eye_data_path)
                all_data = pd.concat([both_eye_data, one_eye_data])
                tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
                tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_40_subjects.pkl")
                fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
                scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
                fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__40_subjects.pkl")
                scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__40_subjects.pkl")

    return stimArray, scanpath_df, fixation_df

def cnn_lstm_model_run(stimArray, scanpath_df):

    seed = 33
    stimType = "Face"
    patch_size = 60
    saliency=False
    is_patch = True
    currpath = os.getcwd()
    run_name = "_cnn_lstm_run_1_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df, stimType)
    df = datasetbuilder.create_patches_dataset(currpath, scanpaths, images, labels, patch_size, saliency)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch)

    cnn_lstm = CnnLstm(seed, split_dataset, saliency, patch_size, run_name)
    # Build train and evaluate model
    cnn_lstm.define_model()
    cnn_lstm.train_model()
    cnn_lstm.metrices(currpath)

def cnn_lstm_img_concat_model_run(stimArray, scanpath_df):
    seed = 33
    stimType = "Face"
    patch_size = 60
    saliency = False
    is_patch = True
    is_simple_lstm = False
    currpath = os.getcwd()
    run_name = "_cnn_lstm_img_concat_run_1_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.create_patches_dataset(currpath, scanpaths, images, labels, patch_size, saliency)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm)

    cnn_lstm_img_concat = CnnLstmImgConcat(seed, split_dataset, saliency, patch_size, run_name, stim_size)
    # Build train and evaluate model
    cnn_lstm_img_concat.define_model()
    cnn_lstm_img_concat.train_model()
    cnn_lstm_img_concat.metrices(currpath)

def simple_lstm_model_run(stimArray, scanpath_df):
    seed = 33
    stimType = "Face"
    is_patch = False
    is_simple_lstm = True
    currpath = os.getcwd()
    run_name = "_simple_lstm_run_1_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.get_scanpath_for_simple_lstm(scanpaths, images, labels)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm)

    simple_lstm = SimpleLstm(seed, split_dataset, run_name)
    # Build train and evaluate model
    simple_lstm.define_model()
    simple_lstm.train_model()
    simple_lstm.metrices(currpath)

def cnn_multi_input_model_run(stimArray, fixation_df):
    seed = 33
    stimType = "Face"
    is_patch = False
    is_simple_lstm = False
    saliency = False
    is_colored_path = True
    currpath = os.getcwd()
    run_name = "vggnet_cnn_multi_input_run_biggerLR_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    maps, images, labels, stim_size = datasetbuilder.load_fixations_related_datasets(currpath, fixation_df,
                                                                                         stimType)
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)

    df = datasetbuilder.get_time_colored_dataset(scanpaths, maps, images, labels, stimType)

    #df = datasetbuilder.get_fixations_for_cnn(scanpaths, maps, images, labels)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)
    cnn_multi_input = CnnMultiInput(seed, split_dataset, saliency, run_name, stim_size)
    # Build train and evaluate model
    cnn_multi_input.define_model()
    cnn_multi_input.train_model()
    cnn_multi_input.metrices(currpath)

def cnn_multi_input_vggnet_model_run(stimArray, fixation_df):
    seed = 33
    stimType = "Face"
    is_patch = False
    is_simple_lstm = False
    saliency = False
    currpath = os.getcwd()
    run_name = "_simple_lstm_run_1_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    maps, images, labels, stim_size = datasetbuilder.load_fixations_related_datasets(currpath, fixation_df,
                                                                                     stimType)
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.get_fixations_for_cnn(scanpaths, maps, images, labels)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm)

    cnn_multi_input_vggnet = CnnMultiInputVGGNet(seed, split_dataset, saliency, run_name, stim_size)
    # Build train and evaluate model
    cnn_multi_input_vggnet.define_model()
    cnn_multi_input_vggnet.train_model()
    cnn_multi_input_vggnet.metrices(currpath)

def svm_run(stimArray, scanpath_df):
    seed = 33
    stimType = "Face"
    currpath = os.getcwd()
    run_name = "_simple_lstm_run_1_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.get_scanpath_for_simple_lstm(scanpaths, images, labels)

    svm = SVM(seed, df, run_name)
    # Build train and evaluate model
    svm.run_model()


stimArray, scanpath_df, fixation_df = get_datasets()
#stimArray, scanpath_df, fixation_df, colored_df = get_datasets()
#cnn_multi_input_model_run(stimArray, colored_df)
cnn_multi_input_model_run(stimArray, fixation_df)

"""
def main():
    stimArray, scanpath_df, fixation_df = get_datasets()
    cnn_multi_input_model_run(stimArray, fixation_df)
    # simple_lstm_model_run(stimArray, scanpath_df)
    # cnn_lstm_model_run(stimArray, scanpath_df)
    # svm_run(stimArray, scanpath_df)


if __name__ == '__main__':
    main()
"""