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
from modules.models.cnn_stacked_frames_input import CnnStackedFrames
from modules.models.cnn_stacked_frames_image_concat import CnnStackedFramesImageConcat
from modules.models.binary_simple_cnn import BinarySimpleCnn
from modules.models.binary_two_stream_cnn import BinaryTwoStreamCnn
import os
import yaml
from datetime import datetime

def get_datasets(x_subjects):
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
        fixation_df = pd.read_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
        scanpath_df = pd.read_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")
    except:
        try:
            print("Log... reading tidy df")
            tidy_data = pd.read_pickle(path + "/etp_data/processed/tidy_data_" + x_subjects + "_subjects.pkl")
            fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
            scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
            fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
            scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")
        except:
            try:
                print("Log... reading raw data csv")
                both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
                one_eye_data = pd.read_csv(path + "/etp_data/processed/" + x_subjects + "_subjects_one_eye_data.csv") #(one_eye_data_path)
                all_data = pd.concat([both_eye_data, one_eye_data])
                tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
                tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_" + x_subjects + "_subjects.pkl")
                fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
                scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
                fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
                scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")
            except:
                print("Log... processing raw data to csv")
                both_eye_data_path = data.read_eyeTracking_data_both_eye_recorded(fixation_only)
                one_eye_data_path = data.read_eyeTracking_data_one_eye_recorded(fixation_only)
                both_eye_data = pd.read_csv(path + "/etp_data/processed/40_subjects_both_eye_data.csv") #(both_eye_data_path)
                one_eye_data = pd.read_csv(path + "/etp_data/processed/" + x_subjects + "_subjects_one_eye_data.csv") #(one_eye_data_path)
                all_data = pd.concat([both_eye_data, one_eye_data])
                tidy_data = data.data_tidying_for_dataset_building(all_data, cfg['exp']['etp']['screen_resolution'])
                tidy_data.to_pickle(path + "/etp_data/processed/tidy_data_" + x_subjects + "_subjects.pkl")
                fixation_df = datasetbuilder.get_fixation_dataset(tidy_data)
                scanpath_df = datasetbuilder.get_scanpath_dataset(tidy_data)
                fixation_df.to_pickle(path + "/etp_data/processed/fixation_df__" + x_subjects + "_subjects.pkl")
                scanpath_df.to_pickle(path + "/etp_data/processed/scanpath_df__" + x_subjects + "_subjects.pkl")

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
    is_colored_path = False
    currpath = os.getcwd()
    run_name = "_cnn_lstm_vggnet_patchImage_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.create_patches_dataset(currpath, scanpaths, images, labels, patch_size, saliency)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)

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

def cnn_multi_input_model_run(stimArray, fixation_df, scanpath_df):
    seed = 33
    stimType = "Face"
    is_patch = False
    is_simple_lstm = False
    saliency = False
    is_colored_path = True
    timePeriodMilisec = 0
    currpath = os.getcwd()
    run_name = "vggNet_biggerBatch64_" + stimType
    run_number = datetime.now()
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    maps, images, labels, stim_size = datasetbuilder.load_fixations_related_datasets(currpath, fixation_df,
                                                                                         stimType)
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)

    df = datasetbuilder.get_time_colored_dataset(scanpaths, maps, images, labels, stimType, timePeriodMilisec)
    #df = datasetbuilder.get_fixations_for_cnn(scanpaths, maps, images, labels)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)
    cnn_multi_input = CnnMultiInput(seed, split_dataset, saliency, run_name, stim_size, run_number)
    # Build train and evaluate model
    cnn_multi_input.define_model()
    cnn_multi_input.train_model()
    cnn_multi_input.metrices(currpath)

def cnn_stacked_frames_input_model_run(stimArray, scanpath_df):
    seed = 33
    stimType = "Face"
    patch_size = 60
    num_patches = 30
    is_patch = True
    is_simple_lstm = False
    saliency = False
    is_colored_path = False
    currpath = os.getcwd()
    run_name = "vgg16_stacked_frames_only" + stimType
    run_number = datetime.now()
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)

    df = datasetbuilder.create_patches_dataset(currpath, scanpaths, images, labels, num_patches, patch_size, saliency)
    df = datasetbuilder.create_stacked_frames_dataset(df)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)
    cnn_stacked_frames = CnnStackedFrames(seed, split_dataset, saliency, run_name, patch_size, run_number, num_patches)
    # Build train and evaluate model
    cnn_stacked_frames.define_model()
    cnn_stacked_frames.train_model()
    cnn_stacked_frames.metrices(currpath)

def cnn_stacked_frames_image_concat_input_model_run(stimArray, scanpath_df):
    seed = 33
    stimType = "Face"
    patch_size = 60
    num_patches = 30
    is_patch = True
    is_simple_lstm = False
    saliency = False
    is_colored_path = False
    currpath = os.getcwd()
    run_name = "cnn_stacked_frames_vggNetImage_concat_" + stimType
    run_number = datetime.now()
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)

    df = datasetbuilder.create_patches_dataset(currpath, scanpaths, images, labels, num_patches, patch_size, saliency)
    df = datasetbuilder.create_stacked_frames_dataset(df)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)
    cnn_stacked_frames = CnnStackedFramesImageConcat(seed, split_dataset, saliency, run_name, stim_size, patch_size, run_number, num_patches)
    # Build train and evaluate model
    cnn_stacked_frames.define_model()
    cnn_stacked_frames.train_model()
    cnn_stacked_frames.metrices(currpath)


def svm_run(stimArray, scanpath_df):
    seed = 33
    stimType = "Face"
    currpath = os.getcwd()
    run_name = "svm_run_" + stimType
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.get_scanpath_for_simple_lstm(scanpaths, images, labels)

    svm = SVM(seed, df, run_name)
    # Build train and evaluate model
    svm.run_model()

def binary_simple_cnn_run(stimArray, fixation_df, scanpath_df):
    seed = 33
    stimType = "Face"
    is_patch = False
    is_simple_lstm = False
    saliency = False
    is_colored_path = True
    timePeriodMilisec = 100
    currpath = os.getcwd()
    run_name = "binary_simple_cnn_" + stimType
    run_number = datetime.now()
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    maps, images, labels, stim_size = datasetbuilder.load_fixations_related_datasets(currpath, fixation_df,
                                                                                         stimType)
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    df = datasetbuilder.get_time_colored_dataset(scanpaths, maps, images, labels, stimType, timePeriodMilisec)
    #df = datasetbuilder.get_fixations_for_cnn(scanpaths, maps, images, labels)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)
    binary_simple_cnn = BinarySimpleCnn(seed, split_dataset, saliency, run_name, stim_size, run_number)
    # Build train and evaluate model
    binary_simple_cnn.define_model()
    binary_simple_cnn.train_model()
    binary_simple_cnn.metrices(currpath)

def binary_two_stream_run(stimArray, fixation_df, scanpath_df):
    seed = 33
    stimType = "Face"
    is_patch = False
    is_simple_lstm = False
    saliency = False
    is_colored_path = False
    timePeriodMilisec = 100
    currpath = os.getcwd()
    run_name = "binary_simple_cnn_" + stimType
    run_number = datetime.now()
    datasetbuilder = DatasetBuilder([stimArray[0], stimArray[1]])
    maps, images, labels, stim_size = datasetbuilder.load_fixations_related_datasets(currpath, fixation_df,
                                                                                         stimType)
    scanpaths, images, labels, stim_size = datasetbuilder.load_scanpath_related_datasets(currpath, scanpath_df,
                                                                                         stimType)
    #df = datasetbuilder.get_time_colored_dataset(scanpaths, maps, images, labels, stimType, timePeriodMilisec)
    df = datasetbuilder.get_fixations_for_cnn(scanpaths, maps, images, labels)
    split_dataset = datasetbuilder.train_test_val_split_stratify_by_subject(df, seed, is_patch, is_simple_lstm, is_colored_path)
    binary_two_stream_cnn = BinaryTwoStreamCnn(seed, split_dataset, saliency, run_name, stim_size, run_number)
    # Build train and evaluate model
    binary_two_stream_cnn.define_model()
    binary_two_stream_cnn.train_model()
    binary_two_stream_cnn.metrices(currpath)


stimArray, scanpath_df_old, fixation_df_old = get_datasets("40")
stimArray, scanpath_df_new, fixation_df_new = get_datasets("new")
fixation_df = pd.concat([fixation_df_old, fixation_df_new])
scanpath_df = pd.concat([scanpath_df_old, scanpath_df_new])
binary_two_stream_run(stimArray, fixation_df, scanpath_df)
#binary_simple_cnn_run(stimArray, fixation_df, scanpath_df)
#cnn_multi_input_model_run(stimArray, fixation_df, scanpath_df)
#cnn_stacked_frames_input_model_run(stimArray, scanpath_df)
#cnn_stacked_frames_image_concat_input_model_run(stimArray, scanpath_df)
"""
def main():
    stimArray, scanpath_df, fixation_df = get_datasets()
    cnn_multi_input_model_run(stimArray, fixation_df)


if __name__ == '__main__':
    main()
"""