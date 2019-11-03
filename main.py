import visualize_data_functions as vis
import run_model
import sys
sys.path.append('../')
import models
import ds_readers as ds
import matplotlib as mpl

"""
Visualization lalala
"""

fixation_df = ds.get_fixation_df()
scanpath_df = ds.get_scanpath_df()
stimTypes = fixation_df.stimType.unique()
vis.visualize(fixation_df, scanpath_df, stimTypes[0])

"""
num_epochs = 10
batch_size = 64

""""""Run model CNN simple""""""
model = models.cnnSimple()

use_cuda, log, x_dtype, y_dtype, n_epochs, criterion = run_model.init(num_epochs, "cnn_loss.txt")

print("preparing training and test sets")
x_train, x_test, y_train, y_test = ds.get_train_test_dataset()

X_train, X_val, Y_train, Y_val = ds.get_train_val_dataset(x_train, y_train)

print("Start train")
train_loss_curve, val_loss_curve = run_model.train_by_batches(model, X_train, Y_train, X_val, Y_val, criterion, use_cuda, log, x_dtype, y_dtype, n_epochs, batch_size)

print("Run test")
test_total_loss = run_model.test(model, x_test, y_test, criterion, log, x_dtype, y_dtype)

log.close()

#torch.save(model.state_dict(), models.checkpoint_cnn_path())

plot = run_model.plot_train_val_loss_curve(use_cuda, n_epochs, train_loss_curve, val_loss_curve)

plot.show()
"""