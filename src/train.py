import numpy as np 
import torch 
from models.dil_resnet import DRN
from models.FNO_3d_time import FNO3d
from utils.Adam import Adam
import os
import logging
from datetime import datetime 
from utils.utilities3 import LpLoss
from utils.utils import visualize_training_curves, get_train_test_loaders, visualize_training_curves_40

"""
train a model 
"""

def train_model(yaml_data, args):
    now = datetime.now() 
    logging.basicConfig(level=logging.DEBUG, filename=f"../logs/sample_debug_{now}.log")
    logging.basicConfig(level=logging.INFO, filename=f"../logs/sample_info_{now}.log")
    train_loader, test_loader = get_train_test_loaders(yaml_data)
    device = torch.device("cuda:7")
    learning_rate = yaml_data["training"]["learning_rate"]
    scheduler_step = yaml_data["training"]["step_size"]
    scheduler_gamma = yaml_data["training"]["gamma"]
    epochs = yaml_data["training"]["epochs"]
    batch_size = yaml_data["training"]["batch_size"]

    model_name = args.model

    if model_name == "FNO":
        modes = yaml_data[model_name]["modes"]
        width = yaml_data[model_name]["width"]
        model = FNO3d(modes, modes, modes, width).to(device)
        model_file_name = f"FNO3d_modes={modes}_width={width}.pt"

    else:
        in_channels = yaml_data[model_name]["in_channels"]
        out_channels_lst = yaml_data[model_name]["out_channels_lst"]
        kernel_size = yaml_data[model_name]["kernel_size"]
        model = DRN(in_channels=in_channels, out_channels_lst=out_channels_lst, kernel_size=kernel_size).to(device)
        model_file_name = f"drResNet_in={in_channels}_out={out_channels_lst}_ker={kernel_size}.pt"
    
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
    myloss = LpLoss(size_average=False)


    train_losses = [] 
    test_losses = [] 
    for ep in range(epochs):
        model.train()
        train_losses_batch = [] 
        train_l2_step = 0
        for input, output in train_loader:
            input = input.to(device)
            output = output.to(device)
            pred = model(input)
            # print(f"minibatch: input = {input.shape}, pred = {pred.shape}, output={output.shape}")
            loss = myloss(pred.reshape(batch_size, -1), output.reshape(batch_size, -1))
            train_l2_step += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_losses_batch.append(loss.item())
        avg_train_loss = np.mean(np.array(train_losses_batch))
        logging.info(f"epoch = {ep}, avg_train_loss = {avg_train_loss}")
        train_losses.append(avg_train_loss)

        test_losses_batch = [] 
        with torch.no_grad():
            for input, output in test_loader:
                input = input.to(device)
                output = output.to(device)
                pred = model(input)
                loss = myloss(pred.reshape(batch_size, -1), output.reshape(batch_size, -1))
                test_losses_batch.append(loss.item())
        avg_test_loss = np.mean(np.array(test_losses_batch))
        logging.info(f"epoch = {ep}, avg_test_loss = {avg_test_loss}")
        test_losses.append(avg_test_loss)
        scheduler.step()

    model_path = yaml_data[model_name]["path"]
    torch.save(model.state_dict(), os.path.join(model_path, model_file_name))
    visualize_training_curves(tr_losses=train_losses, te_losses=test_losses, model_name=model_name)
    visualize_training_curves_40(tr_losses=train_losses, te_losses=test_losses, model_name=model_name)