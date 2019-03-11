import visualize_data_functions as vis
import run_model
import sys
sys.path.append('../')
import models
import ds_readers as ds
import matplotlib as mpl

"""
Visualization
"""
fixation_df = ds.get_fixation_df()
scanpath_df = ds.get_scanpath_df()
stimTypes = fixation_df.stimType.unique()
vis.visualize(fixation_df, scanpath_df, stimTypes[0])


"""Run model CNN simple"""
use_cuda, log, x_dtype, y_dtype, n_epochs, criterion = run_model.init(10, "cnn_loss.txt")

#chose the model to run
model = models.cnnSimple()

print("preparing training and test sets")
x_train, x_test, y_train, y_test = ds.get_train_test_dataset()

X_train, X_val, Y_train, Y_val = ds.get_train_val_dataset(x_train, y_train)

print("Start train")
train_loss_curve, val_loss_curve = run_model.train(model, X_train, Y_train, X_val, Y_val, criterion, use_cuda, log, x_dtype, y_dtype, n_epochs)

print("Run test")
test_total_loss = run_model.test(model, x_test, y_test, criterion, log, x_dtype, y_dtype)

log.close()

#torch.save(model.state_dict(), models.checkpoint_cnn_path())

if use_cuda:
    mpl.use('Agg')
    import matplotlib.pyplot as plt

    plt.ioff()  # http://matplotlib.org/faq/usage_faq.html (interactive mode)
else:
    import matplotlib.pyplot as plt

plt.title('Loss function value for validation and train after each epoch')
plt.xlabel('epoch')
plt.ylabel('loss')

epochs = list(range(1, n_epochs + 1))
plt.plot(epochs, train_loss_curve, 'b', label='Q1 Train Data')
plt.plot(epochs, val_loss_curve, 'r', label='Q1 Validation Data')

plt.legend(loc='best')
plt.savefig('cnn_train_val_plot.png')