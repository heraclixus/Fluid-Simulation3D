import matplotlib.pyplot as plt 
from numpy import arange
from datetime import datetime
import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset


def visualize_training_curves(tr_losses, te_losses):
    now = datetime.now()
    epochs = range(1, len(tr_losses) + 1) 
    plt.plot(epochs, tr_losses, label="Training loss")
    plt.plot(epochs, te_losses, label="Test loss")
    plt.title("Training vs. Test loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(arange(1, len(tr_losses)+1, 10))
    plt.legend(loc="best")
    plt.savefig(f"figs/training_vis_{now}.png")    


"""
read from npz file and return data loaders
"""
def get_train_test_loaders(yaml_data):
    data = np.load(yaml_data["dataset"]["path"])
    data_inputs, data_outputs = data["inputs"][:,:,:,:-1,:], data["outputs"]
    n_samples = len(data_inputs)
    idx = torch.randperm(n_samples)
    train_size = int(n_samples * 0.8)    
    train_indices = idx[:train_size]
    test_indices = idx[train_size:]

    # FNO expects the input to be (batch_size, x,y, z, c)
    data_inputs = torch.tensor(data_inputs).permute(0, 2, 3, 4, 1)
    data_outputs = torch.tensor(data_outputs).permute(0, 2, 3, 4, 1)
    x_train, x_test = data_inputs[train_indices], data_inputs[test_indices]
    y_train, y_test = data_outputs[train_indices], data_outputs[test_indices]

    print(f"x_train shape = {x_train.shape}, y_train shape = {y_train.shape}")
    print(f"x_test shape = {x_test.shape}, y_test shape = {y_test.shape}")

    train_loader = DataLoader(TensorDataset(x_train, y_train), 
                                batch_size=yaml_data["training"]["batch_size"], shuffle=True, drop_last=True)
    test_loader = DataLoader(TensorDataset(x_test, y_test), 
                                batch_size=yaml_data["training"]["batch_size"], shuffle=True, drop_last=True)
    return train_loader, test_loader



"""
visualize the ground truth output scalar field against the predicted one
"""
def visualize_predictions(pred, index, output, var=0, axis=2, interpolation="bilinear"):
    plot_slices(pred[index], type="prediction", var=var, axis=axis, interpolation=interpolation)
    plot_slices(output[index], type="output", var=var, axis=axis, interpolation=interpolation)    


"""
helper function provided
"""
def plot_slices(data, type, var=0, axis=2, interpolation='bilinear'):
    """Visualize a stack of images for Assignment 1.
    
    Parameters
    -----------
    data : array-like, 4d (U, x, y, z)
        The data to plot
    type: str
        pred or output data
    var : int
        The variable in U to plot, 0 -> Ux, 1 -> Uy, 2 -> Uz.
    axis: int
        The axis to slice down. The simulation is pseudo-3d meaning
        axis=2 gives the most physically realistic looking profiles.
        Focus on this axis in analyzing your model's results.
    interpolation: str, default: rcParams["image.interpolation"] (default: 'antialiased')
        The interpolation method used.
        Supported values are 'none', 'antialiased', 'nearest', 'bilinear', 'bicubic', 'spline16', 
        'spline36', 'hanning', 'hamming', 'hermite', 'kaiser', 'quadric', 'catrom', 'gaussian', 
        'bessel', 'mitchell', 'sinc', 'lanczos', 'blackman'.
    """
    
    fig, ax = plt.subplots(2, 5, figsize=(7, 3.5), sharex="all")

    # Nice major title
    if var == 0:
        var_code = "$U_x$"
    if var == 1:
        var_code = "$U_y$"
    if var == 2:
        var_code = "$U_z$"
    plt.suptitle(var_code)

    # Plot slice + nice subtitle.
    for i in range(10):
        if axis == 0:
            ax.ravel()[i].imshow(data[var, i, :, :], interpolation=interpolation)
            ax.ravel()[i].set_title(f"$x={i}$")
        elif axis == 1:
            ax.ravel()[i].imshow(data[var, :, i, :], interpolation=interpolation)
            ax.ravel()[i].set_title(f"$y={i}$")
        elif axis == 2:
            ax.ravel()[i].imshow(data[var, :, :, i], interpolation=interpolation)
            ax.ravel()[i].set_title(f"$z={i}$")
        else:
            raise ValueError("slice_axis must be between [0, 2]")
        
        # No axis labels
        ax.ravel()[i].get_xaxis().set_visible(False)
        ax.ravel()[i].get_yaxis().set_visible(False)
    plt.savefig(f"figs/scalar_field_type={type}_var={var}_axis={axis}.png")