from modules.data.datasets import DatasetBuilder


seed = 33
datasetbuilder = DatasetBuilder()
stims_array, scanpath_df, fixation_df = datasetbuilder.processed_data_loader()
stimType = "Face"
train, val, test = datasetbuilder.train_test_val_split(stimType, scanpath_df, fixation_df, seed)
train.reset_index(inplace=True)
val.reset_index(inplace=True)
test.reset_index(inplace=True)
print("Log... Train shape", train.shape)
print("Log... Val shape", val.shape)
print("Log... Test shape", test.shape)


train = datasetbuilder.preper_data_for_model(train, stimType, is_scanpath=True, is_fixation=False,
                                             is_coloredpath=False, color_split=110, is_img=False)
val = datasetbuilder.preper_data_for_model(train, stimType, is_scanpath=True, is_fixation=False,
                                             is_coloredpath=False, color_split=110, is_img=False)
test = datasetbuilder.preper_data_for_model(train, stimType, is_scanpath=True, is_fixation=False,
                                             is_coloredpath=False, color_split=110, is_img=False)




binary_two_stream_cnn = BinaryTwoStreamCnn(seed, split_dataset, saliency, run_name, stim_size, run_number)
# Build train and evaluate model
binary_two_stream_cnn.define_model()
binary_two_stream_cnn.train_model()
binary_two_stream_cnn.metrices(currpath)
