import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.cityscapes import Cityscapes
from config.default import get_cfg_defaults
from models.hrnet import HRNetV2
import argparse
import utils.transforms as t
from utils.utils import Denormalize
from torch_poly_lr_decay import PolynomialLRDecay
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
import os
from tqdm import tqdm
from sync_batchnorm import convert_model, DataParallelWithCallback

def main(args):
    cfg = load_config(args.config_path)
    device_print = 'cuda:'+ ",".join(map(str,cfg.CUDA.CUDA_NUM)) if cfg.CUDA.USE_CUDA and torch.cuda.is_available() else 'cpu'
    print("device:"+str(device_print))
    base_device = torch.device('cuda:'+ str(cfg.CUDA.CUDA_NUM[0]) if cfg.CUDA.USE_CUDA and torch.cuda.is_available() else 'cpu')
    
    # prepare dataset
    train_transform = t.PairCompose([
        t.PairRandomScale(scale_range=(0.5, 2.0)),
        t.PairRandomCrop(size=(512, 1024), pad_if_needed=True),
        t.PairRandomHorizontalFlip(p=0.5),
        t.PairToTensor(),
        t.PairNormalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])
    ])
    train_dataset = Cityscapes(root=cfg.DATASET.ROOT, split='train', target_type='semantic', transform=train_transform)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True, num_workers=cfg.TRAIN.NUM_WORKERS, drop_last=True)
    denorm = Denormalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
    
    # model
    print("setting up model ...")
    model = HRNetV2(40, 19)
    model = nn.DataParallel(model, device_ids=cfg.CUDA.CUDA_NUM)
    model.to(base_device)
    if cfg.TRAIN.CHECKPOINT != '':
        model.load_state_dict(torch.load(cfg.TRAIN.CHECKPOINT))
    
    # checkpoint path
    if not os.path.exists(cfg.TRAIN.SAVE_WEIGHT_PATH):
        os.makedirs(cfg.TRAIN.SAVE_WEIGHT_PATH)
        
    # optimizer
    optimizer = torch.optim.SGD(model.parameters(),lr=cfg.TRAIN.LERNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = PolynomialLRDecay(optimizer, max_decay_steps=100, end_learning_rate=0.001 power=0.9)
    
    # loss
    loss = nn.CrossEntropyLoss(ignore_index=255, reduction='mean')
    
    # prepare logging
    if not os.path.exists(cfg.TRAIN.LOG_PATH):
        os.makedirs(cfg.TRAIN.LOG_PATH)
    writer = SummaryWriter(cfg.TRAIN.LOG_PATH)
    
    print("=================start training==================")
    
    for epoch in range(cfg.TRAIN.EPOCH_START, cfg.TRAIN.EPOCH_END + 1):
        print(f'epoch : {epoch}')
        for i , (image, label) in enumerate(tqdm(train_dataloader)):
            model.train()
            step = epoch * len(train_dataloader) + i
            
            image = image.to(base_device, dtype=torch.float32)
            label = label.to(base_device, dtype=torch.long)
            
            optimizer.zero_grad()
            
            pred = model(image)
            loss_pred = loss(pred, label)
            
            # back prop
            loss_pred.backward()
            optimizer.step()
            
            if (i+1) % cfg.TRAIN.LOG_LOSS == 0:
                np_loss = loss_pred.detach().cpu().numpy()
                writer.add_scalar('loss', np_loss, step)
                
            if (i+1) % cfg.TRAIN.LOG_IMAGE == 0:
                image_save = denorm(image[0])
                target_save = label[0].detach().cpu().numpy()
                pred_save = pred.detach().max(dim=1)[1].cpu().numpy()[0]
                #image_save = (denorm(image_save) * 255).transpose(1, 2, 0).astype(np.uint8)
                target_save = train_dataloader.dataset.decode_target(target_save).astype(np.uint8)
                target_save = torch.from_numpy(target_save.astype(np.float32)).clone().permute(2, 0, 1)
                pred_save = train_dataloader.dataset.decode_target(pred_save).astype(np.uint8)
                pred_save = torch.from_numpy(pred_save.astype(np.float32)).clone().permute(2, 0, 1)
                writer.add_image('train_image', image_save, step)
                writer.add_image('label_image', target_save, step)
                writer.add_image('pred_image', pred_save, step)
                
            del image, label, pred
            
            if (step + 1) % cfg.TRAIN.SAVE_WEIGHT_STEP == 0:
                torch.save(model.state_dict(), cfg.TRAIN.SAVE_WEIGHT_PATH + f'/checkpoint_epoch{epoch}_iter{step}.pth')
            
        torch.save(model.module.state_dict(), cfg.TRAIN.SAVE_WEIGHT_PATH + f'/checkpoint_epoch{epoch}_final.pth')
        scheduler.step()
            
    """
    for i in range(0, 1):
        image_save = image[i].cpu().numpy()
        target_save = target[i].cpu().numpy()
        image_save = (denorm(image_save) * 255).transpose(1, 2, 0).astype(np.uint8)
        target_save = train_dataloader.dataset.decode_target(target_save).astype(np.uint8)
    
        Image.fromarray(image_save).save('image_test_{}.png'.format(i))
        Image.fromarray(target_save).save('target_test_{}.png'.format(i))
    image, target = tmp.next()
    for i in range(1, 2):
        image_save = image[i].cpu().numpy()
        target_save = target[i].cpu().numpy()
        image_save = (denorm(image_save) * 255).transpose(1, 2, 0).astype(np.uint8)
        target_save = train_dataloader.dataset.decode_target(target_save).astype(np.uint8)
        
        Image.fromarray(image_save).save('image_test_{}.png'.format(i))
        Image.fromarray(target_save).save('target_test_{}.png'.format(i))
    
    """
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