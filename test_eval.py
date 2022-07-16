import torch
import argparse
from utils import transforms as t
from utils.utils import Denormalize
from dataset.cityscapes import Cityscapes
from models.hrnet import HRNetV2
from torch.utils.data import DataLoader
from config.default import get_cfg_defaults
from metrics.metrics import SegMetrics
from tqdm import tqdm
import numpy as np
from PIL import Image
import os


def main(args):
    cfg = load_config("./config/test_eval.yaml")
    #cfg = load_config(args.config_path)
    device = torch.device('cuda:'+str(cfg.CUDA.CUDA_NUM[0]) if cfg.CUDA.USE_CUDA and torch.cuda.is_available() else 'cpu')
    print("device:"+str(device))
    
    # prepare dataset
    test_transform = t.PairCompose([
        #t.PairRandomCrop(size=(512, 1024)),
        t.PairToTensor(),
        t.PairNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    test_dataset = Cityscapes(root=cfg.DATASET.ROOT, split='val', target_type='semantic', transform=test_transform)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.TEST.BATCH_SIZE, shuffle=True, num_workers=cfg.TEST.NUM_WORKERS, drop_last=True)
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    
    # model
    print("setting up model ...")
    model = HRNetV2(40, 19)
    model.to(device)
    if cfg.TEST.CHECKPOINT != '':
        print("loading pretrained model ...")
        model.load_state_dict(torch.load(cfg.TEST.CHECKPOINT))
        
    # metric
    metric = SegMetrics(19, device)
    
    # results path
    if cfg.TEST.RESULTS_NUM != 0 and not os.path.exists(cfg.TEST.RESULTS_PATH):
        os.makedirs(cfg.TEST.RESULTS_PATH)
    save_count = 0
    
    model.eval()
        
    for i , (image, target) in enumerate(tqdm(test_dataloader)):
        with torch.no_grad():
            image = image.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)
            
            output = model(image)
            pred = output.detach().max(dim=1)[1]
            
            metric.update(target, pred)
            
            if cfg.TEST.RESULTS_NUM != 0 and i % (len(test_dataloader) // cfg.TEST.RESULTS_NUM) == 0:
                image_save = image[0].detach().cpu().numpy()
                target_save = target[0].cpu().numpy()
                pred_save = pred[0].cpu().numpy()
                
                image_save = (denorm(image_save)*255).transpose(1, 2, 0).astype(np.uint8)
                target_save = test_dataloader.dataset.decode_target(target_save).astype(np.uint8)
                pred_save = test_dataloader.dataset.decode_target(pred_save).astype(np.uint8)
                
                Image.fromarray(image_save).save(cfg.TEST.RESULTS_PATH+"/image_{}.png".format(save_count))
                Image.fromarray(target_save).save(cfg.TEST.RESULTS_PATH+"/label_{}.png".format(save_count))
                Image.fromarray(pred_save).save(cfg.TEST.RESULTS_PATH+"/predict_{}.png".format(save_count))
                
                save_count += 1
    
    score = metric.get_results()
    
    print(score)
            
    
def load_config(config_path=None):
    cfg = get_cfg_defaults()
    if config_path is not None:
        cfg.merge_from_file(config_path)
    cfg.freeze()
    return cfg

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--config_path", type=str, help="extra config file path", default=None)
    
    args = parser.parse_args()
    
    main(args)