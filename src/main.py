import numpy as np 
import yaml 
import os
from utils.utils import visualize_predictions
from train import train_model
from models.FNO_3d_time import FNO3d
from models.dil_resnet import DRN
import torch
import argparse


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="choice of models from [FNO, drResNet]", default="FNO")
    parser.add_argument("--device",  type=str, help="device cpu or cuda", default="cuda")    
    args = parser.parse_args()
    
    device = args.device
    model = args.model
    
    with open(f"../configs/train.yaml") as f:
        yaml_data = yaml.safe_load(f)
    
    mode = yaml_data["main"]["mode"]
    
    if mode == "train":
        train_model(yaml_data, args)
        
    else: # inference
        if args.model == "FNO":
            modes, width = yaml_data[args.model]["modes"], yaml_data[args.model]["width"]
            model = FNO3d(modes, modes, modes, width).to(device)
            model_file_name = f"FNO3d_modes={modes}_width={width}.pt"
            model_path = os.path.join(yaml_data[args.model]["path"], model_file_name)
            save_path = os.path.join(yaml_data["outputs"]["path"], model_file_name)

        else:
            in_channels = yaml_data[args.model]["in_channels"]
            out_channels_lst = yaml_data[args.model]["out_channels_lst"]
            kernel_size = yaml_data[args.model]["kernel_size"]
            model = DRN(in_channels=in_channels, out_channels_lst=out_channels_lst, kernel_size=kernel_size).to(device)
            model_file_name = f"drResNet_in={in_channels}_out={out_channels_lst}_ker={kernel_size}.pt"
            model_path = os.path.join(yaml_data[args.model]["path"], model_file_name)
            save_path = os.path.join(yaml_data["outputs"]["path"], model_file_name)

        model.load_state_dict(torch.load(model_path))
        data = np.load(yaml_data["dataset"]["path"])
        data_inputs = torch.tensor(data["inputs"][:,:,:,:-1,:]).permute(0,2,3,4,1).to(device)
        data_outputs = data["outputs"]
        preds = model(data_inputs)
        preds = preds.permute(0,4,1,2,3).detach().cpu().numpy()
        with open(save_path, "wb") as f:
            np.save(f, preds)
        visualize_predictions(pred=preds, model_name=args.model, index=0, output=data_outputs)        