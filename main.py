import ds_readers as ds
import numpy as np
import visualize_data_functions as vis



fixation_df, scanpath_df = ds.get_datasets_df()

#visualization
#stimTypes = fixation_df.stimType.unique()
#vis.visualize(fixation_df, scanpath_df, stimTypes[0])


x_train, x_test, y_train, y_test = ds.get_train_test_dataset(fixation_df)


y_train_unique, y_train_counts = np.unique(y_train, return_counts=True)
y_test_unique, y_test_counts = np.unique(y_test, return_counts=True)


print('y_train: ',np.asarray((y_train_unique, y_train_counts)).T)
print('y_test: ',np.asarray((y_test_unique, y_test_counts)).T)

print('done')


