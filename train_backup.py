import argparse
from ultralytics import YOLO
import os
import sys
import numpy as np
from typing import Dict, Optional

# Advanced training parameters for ultimate robustness
EPOCHS = 400      # Extended training for thorough learningort argparse
from ultral# Advanced training parameters for ultimate robustness
EPOCHS = 100      # Extended training for thorough learning
MOSAIC = 0.9      # Strong mosaic for environmental adaptation
OPTIMIZER = 'AdamW' # AdamW optimizer for convergence
MOMENTUM = 0.937   # Optimal momentum
LR0 = 0.00015     # Very conservative learning rate
LRF = 0.000005    # Extremely gradual learning rate decay
SINGLE_CLS = False
IMGSZ = 1024      # Maximum resolution for fine details
BATCH = 4         # Small batch for quality
PATIENCE = 60     # Very patient early stopping
WARMUP_EPOCHS = 20 # Extended warmup period
CLOSE_MOSAIC = 350 # Keep mosaic for most of trainingYOLO
import os
import sys
import numpy as np
from typing import Dict, Optional

class EarlyStoppingCallback:
    def __init__(self, patience: int = 15, min_delta: float = 0.0001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = float('-inf')
        self.counter = 0
        self.best_epoch = 0
        self.stopped_epoch = 0
        
    def check_stop(self, current_value: float, epoch: int) -> bool:
        if current_value > self.best_value + self.min_delta:
            self.best_value = current_value
            self.counter = 0
            self.best_epoch = epoch
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = epoch
                return True
        return False

# Advanced training parameters for extreme robustness
EPOCHS = 50      # Extended training for deep learning
MOSAIC = 0.85     # Balanced mosaic for robust detection
OPTIMIZER = 'AdamW' # AdamW optimizer with smart learning rate
MOMENTUM = 0.937   # Optimized momentum for stability
LR0 = 0.0002      # Conservative initial learning rate
LRF = 0.00001     # Very gradual learning rate decay
SINGLE_CLS = False
IMGSZ = 896       # Higher resolution for better detail
BATCH = 4         # Smaller batch for higher quality
PATIENCE = 50     # Extended patience for convergence
WARMUP_EPOCHS = 15 # Long warmup for stability
CLOSE_MOSAIC = 250 # Keep mosaic until late stages

def create_callback(early_stopper: EarlyStoppingCallback):
    def on_train_epoch_end(trainer):
        metrics = trainer.metrics
        map_value = metrics.get('metrics/mAP50-95(B)', 0)
        
        if early_stopper.check_stop(map_value, trainer.epoch):
            print(f"\nEarly stopping triggered! No improvement for {early_stopper.patience} epochs.")
            print(f"Best mAP50-95: {early_stopper.best_value:.4f} at epoch {early_stopper.best_epoch}")
            trainer.epoch = trainer.epochs  # This will stop training
            return
        
        print(f"\nEpoch {trainer.epoch} Performance:")
        print(f"Current mAP50-95: {map_value:.4f}")
        print(f"Best mAP50-95: {early_stopper.best_value:.4f}")
        print(f"Epochs without improvement: {early_stopper.counter}")
    
    return on_train_epoch_end

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    # epochs
    parser.add_argument('--epochs', type=int, default=EPOCHS, help='Number of epochs')
    # mosaic
    parser.add_argument('--mosaic', type=float, default=MOSAIC, help='Mosaic augmentation')
    # optimizer
    parser.add_argument('--optimizer', type=str, default=OPTIMIZER, help='Optimizer')
    # momentum
    parser.add_argument('--momentum', type=float, default=MOMENTUM, help='Momentum')
    # lr0
    parser.add_argument('--lr0', type=float, default=LR0, help='Initial learning rate')
    # lrf
    parser.add_argument('--lrf', type=float, default=LRF, help='Final learning rate')
    # single_cls
    parser.add_argument('--single_cls', type=bool, default=SINGLE_CLS, help='Single class training')
    args = parser.parse_args()
    
    # Initialize early stopping
    early_stopper = EarlyStoppingCallback(patience=PATIENCE)
    
    this_dir = os.path.dirname(__file__)
    os.chdir(this_dir)
    model = YOLO(os.path.join(this_dir, "yolov8s.pt"))
    # Enhanced training configuration with focus on robust detection
    # Start training with early stopping and optimized parameters
    results = model.train(
        data=os.path.join(this_dir, "yolo_params.yaml"),
        epochs=args.epochs,
        device=0,  # Use GPU
        single_cls=args.single_cls,
        mosaic=args.mosaic,
        optimizer=args.optimizer,
        lr0=args.lr0,
        lrf=args.lrf,
        momentum=args.momentum,
        imgsz=IMGSZ,
        batch=BATCH,
        warmup_epochs=WARMUP_EPOCHS,
        close_mosaic=CLOSE_MOSAIC,
        # Enhanced augmentation pipeline for extreme robustness
        hsv_h=0.035,      # Increased hue variation for lighting
        hsv_s=0.9,        # Maximum saturation variation
        hsv_v=0.7,        # Extreme brightness variation for dark/light
        degrees=20.0,     # Wider rotation for viewpoint robustness
        translate=0.4,    # More translation for position invariance
        scale=0.6,        # Larger scale variation for distance
        fliplr=0.5,      # Horizontal flip
        flipud=0.3,      # More vertical flips
        mixup=0.25,      # Heavy mixup for generalization
        copy_paste=0.35,  # Increased copy-paste for occlusions
        shear=12.0,      # Enhanced shear for perspective
        perspective=0.0015,# Stronger perspective warping
        
        # Optimized loss configuration for maximum precision
        box=12.0,         # Stronger box loss for precise detection
        cls=2.0,          # Higher class loss for better discrimination
        dfl=3.0,          # Enhanced DFL loss for boundaries
        
        # Advanced training stability and regularization
        overlap_mask=True,
        mask_ratio=6,     # Increased mask ratio
        dropout=0.25,     # Heavier dropout for regularization
        label_smoothing=0.2, # Increased smoothing for robustness
        
        # Checkpoint and performance settings
        workers=4,
        save_period=3,    # Very frequent checkpoints
        exist_ok=True,
        verbose=True,
        cache=True,      # Enable cache for faster training
        
        # Enhanced optimization settings
        cos_lr=True,     # Cosine learning rate schedule
        weight_decay=0.0015, # Strong weight decay for better generalization
        nbs=64,          # Standard nominal batch size
        
        # Advanced training features
        amp=True,        # Mixed precision for stability
        fraction=1.0,    # Use all training data
        
        # Comprehensive validation and monitoring
        val=True,
        plots=True,      # Enable plots for monitoring
        rect=False,      # Disable rectangular training for better accuracy
        multi_scale=True,# Enable multi-scale training
        iou=0.7,        # Higher IoU threshold for better precision
        conf=0.001,     # Lower confidence threshold during training
        augment=True,   # Enable TTA during validation
        project="runs"  # Project directory
    )
'''
Mixup boost val pred but reduces test pred
Mosaic shouldn't be 1.0  
'''


'''
                   from  n    params  module                                       arguments
  0                  -1  1       464  ultralytics.nn.modules.conv.Conv             [3, 16, 3, 2]
  1                  -1  1      4672  ultralytics.nn.modules.conv.Conv             [16, 32, 3, 2]
  2                  -1  1      7360  ultralytics.nn.modules.block.C2f             [32, 32, 1, True]
  3                  -1  1     18560  ultralytics.nn.modules.conv.Conv             [32, 64, 3, 2]
  4                  -1  2     49664  ultralytics.nn.modules.block.C2f             [64, 64, 2, True]
  5                  -1  1     73984  ultralytics.nn.modules.conv.Conv             [64, 128, 3, 2]
  6                  -1  2    197632  ultralytics.nn.modules.block.C2f             [128, 128, 2, True]
  7                  -1  1    295424  ultralytics.nn.modules.conv.Conv             [128, 256, 3, 2]
  8                  -1  1    460288  ultralytics.nn.modules.block.C2f             [256, 256, 1, True]
  9                  -1  1    164608  ultralytics.nn.modules.block.SPPF            [256, 256, 5]
 10                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 11             [-1, 6]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 12                  -1  1    148224  ultralytics.nn.modules.block.C2f             [384, 128, 1]
 13                  -1  1         0  torch.nn.modules.upsampling.Upsample         [None, 2, 'nearest']
 14             [-1, 4]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 15                  -1  1     37248  ultralytics.nn.modules.block.C2f             [192, 64, 1]
 16                  -1  1     36992  ultralytics.nn.modules.conv.Conv             [64, 64, 3, 2]
 17            [-1, 12]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 18                  -1  1    123648  ultralytics.nn.modules.block.C2f             [192, 128, 1]
 19                  -1  1    147712  ultralytics.nn.modules.conv.Conv             [128, 128, 3, 2]
 20             [-1, 9]  1         0  ultralytics.nn.modules.conv.Concat           [1]
 21                  -1  1    493056  ultralytics.nn.modules.block.C2f             [384, 256, 1]
 22        [15, 18, 21]  1    751507  ultralytics.nn.modules.head.Detect           [1, [64, 128, 256]]
Model summary: 225 layers, 3,011,043 parameters, 3,011,027 gradients, 8.2 GFLOPs
'''