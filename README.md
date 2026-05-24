# EfficientSCI
This repo is the implementation of [EDecoupled Sensing Matrices with Swin Transformer for High Speed and Resolution Imaging]

## Training 

```
python tools/train.py configs/EfficientSCI/efficientsci_base.py
```

## Testing EfficientSCI on Grayscale Simulation Dataset 
```
python tools/test.py configs/EfficientSCI/efficientsci_base.py --weights=checkpoints/efficientsci_base.pth
```

