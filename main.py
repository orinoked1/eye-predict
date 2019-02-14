import ds_readers as ds
import visualize_data_functions as vis




fixation_df, scanpath_df = ds.get_datasets_df()

#visualization
stimTypes = fixation_df.stimType.unique()
vis.visualize(fixation_df, scanpath_df, stimTypes[0])


X_train, X_test, y_train, y_test = ds.get_train_test_dataset(fixation_df)

print('done')


