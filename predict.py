import torch
import torchvision.transforms.functional as f
import os
import glob
from models.hrnet import HRNetV2
from config.default import get_cfg_defaults
from PIL import Image
from tqdm import tqdm
from dataset.cityscapes import Cityscapes
import numpy as np
import argparse

def main(args):
    cfg = load_config("./config/predict.yaml")
    #cfg = load_config(args.config_path)
    device = torch.device('cuda:'+str(cfg.CUDA.CUDA_NUM[0]) if cfg.CUDA.USE_CUDA and torch.cuda.is_available() else 'cpu')
    print("device:"+str(device))
    
    if os.path.isdir(args.input_path):
        file_list = [*glob.glob(os.path.join(args.input_path, "**.png"), recursive=True),
                     *glob.glob(os.path.join(args.input_path, "**.jpg"), recursive=True)]
    elif os.path.isfile(args.input_path):
        file_list = glob.glob(args.input_path)
    else:
        print("Please input valid image path.")
        return
    
    # model
    print("setting up model ...")
    model = HRNetV2(cfg.MODEL.C, 19).to(device)
    print("loading pretrained model")
    model.load_state_dict(torch.load(cfg.TEST.CHECKPOINT, map_location = device))
    
    # result path
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
        
    model.eval()
    if len(file_list) == 0:
        print("No image files !!")
        return
    
    with torch.no_grad():
        for file in tqdm(file_list):
            image = Image.open(file).convert("RGB")
            image = f.to_tensor(image)
            image = f.normalize(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            image = image.unsqueeze(0)
            image = image.to(device)
            
            output = model(image)
            pred = output.detach().max(dim=1)[1][0].cpu().numpy()
            
            pred_save = Cityscapes.decode_target(pred).astype(np.uint8)
            file_name = os.path.splitext(os.path.basename(file))[0]
            Image.fromarray(pred_save).save(os.path.join(args.output_path, file_name+"_pred.png"))
            
            del image, output, pred, pred_save
    
    
def load_config(config_path=None):
    cfg = get_cfg_defaults()
    if config_path is not None:
        cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", type=str, help="extra config file path", default=None)
    parser.add_argument("--input_path", type=str, help="image path", required=True)
    parser.add_argument("--output_path", type=str, help="extra config file path", required=True)
    
    args = parser.parse_args()
    
    main(args)