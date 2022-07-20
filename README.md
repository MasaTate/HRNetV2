# HRNetV2 for Semantic Segmentation
## Introduction
This the non-official reimplementation of [HRNetV2 for Semantic Segmentation](https://arxiv.org/abs/1904.04514). ([Official code](https://github.com/HRNet/HRNet-Semantic-Segmentation/tree/pytorch-v1.1))  

Although HRNet-OCR is implemented in official code, I implemented pure HRNetV2 without OCR. 
I evaluated the model only on Cityscapes dataset.  

Please refer to [Qiita article](https://qiita.com/mtsnoopy/items/2f5e47d58751b5dee012) for more imformation about this project. (Japanese)

## Set up
Set up your environment by running
```shell
pip install -r requirements.txt 
```

## Training
 Only Cityscapes dataset is supported.  
 First, download [Cityscapes dataset](https://www.cityscapes-dataset.com/).
 Then, modify `./config/default.py` appropriately.  
 For example, train the HRNet-W40 with a batch size of 12 on 4 GPUs (cuda:0, 1, 2, 3):
 ```
 ...
 ...
 # CUDA related prarams
 _C.CUDA = CN()
 _C.CUDA.USE_CUDA = True
 _C.CUDA.CUDA_NUM = [0,1,2,3]
 ...
 ...
 # model
 _C.MODEL = CN()
 _C.MODEL.C = 40 
 ...
 ...

 _C.TRAIN.BATCH_SIZE = 12
```
After you set config file, then run:
```shell
python train.py
```

## Testing
You can test semantic segmentation on your image file. Modify settings in `./config/default.py` and `./config/predict.yaml` as you like, then run:
```
python predict.py --input_path $INPUT_IMG(or DIR)_PATH --output_path $OUTPUT_DIR_PATH
```


## Evaluation
Since ground truth for test set is not publicly available, I used validation set to evaluate.
Modify settings in `./config/default.py` and `./config/test_eval.yaml` as you like, then you can run:
```
python test_eval.py
```

## Reference
- J. Wang, K. Sun, T. Cheng, B. Jiang, C. Deng, Y. Zhao, D. Liu, Y. Mu, M. Tan, X. Wang, W. Liu and B. Xiao "Deep High-Resolution Representation Learning
for Visual Recognition," TPAMI, vol.43, no.10, pp.3349-3364, Apr 2020. [[ieee](https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=9052469)]

- pytorch-polynominal-lr-decay implementation by @cmpark0126 [[github](https://github.com/cmpark0126/pytorch-polynomial-lr-decay)]

- Syncronized-BatchNorm-PyTorch implementation by @vacancy [[githhub](https://github.com/vacancy/Synchronized-BatchNorm-PyTorch)]
