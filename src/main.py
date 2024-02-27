import numpy as np 
import yaml 
import os
from datetime import datetime
from utils.utils import visualize_predictions
from train import train_model
from models.FNO_3d_time import FNO3d
import torch

if __name__ == "__main__":
    
    device = torch.device("cuda:7") 
    
    with open("../configs/FNO.yaml") as f:
        yaml_data = yaml.safe_load(f)
    
    mode = yaml_data["main"]["mode"]
    
    if mode == "train":
        train_model(yaml_data)
        
    else: # inference
        modes, width = yaml_data["model"]["modes"], yaml_data["model"]["width"]
        model = FNO3d(modes, modes, modes, width).to(device)
        model_path = os.path.join(yaml_data["model"]["path"], f"FNO3d_modes={modes}_width={width}.pt")
        model.load_state_dict(torch.load(model_path)) 
        data = np.load(yaml_data["dataset"]["path"])
        data_inputs = torch.tensor(data["inputs"][:,:,:,:-1,:]).permute(0,2,3,4,1).to(device)
        data_outputs = data["outputs"]
        preds = model(data_inputs)
        preds = preds.permute(0,4,1,2,3).detach().cpu().numpy()
        save_path = os.path.join(yaml_data["outputs"]["path"], f"pred_FNO3d_modes={modes}_width={width}.npy")
        with open(save_path, "wb") as f:
            np.save(f, preds)
        visualize_predictions(pred=preds, index=0, output=data_outputs)