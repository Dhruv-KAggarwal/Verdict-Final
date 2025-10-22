from ultralytics import YOLO
import os
import numpy as np
from tabulate import tabulate
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from pathlib import Path
import pandas as pd
from sklearn.metrics import confusion_matrix
import yaml

class ModelEvaluator:
    def __init__(self, model_path='runs/detect/train/weights/best.pt'):
        """Initialize the evaluator with a model path"""
        self.model = YOLO(model_path)
        self.class_names = None
        self.evaluation_dir = Path("evaluation/results")
        self.evaluation_dir.mkdir(parents=True, exist_ok=True)
        
    def load_ground_truth(self, label_path):
        """Load ground truth labels from a file"""
        if not os.path.exists(label_path):
            return []
        with open(label_path, 'r') as f:
            lines = f.readlines()
        return [line.strip().split() for line in lines]

def evaluate_single_prediction(image_path, model_path='runs/detect/train/weights/best.pt'):
    """Evaluate prediction against ground truth for a single image"""
    try:
        # Create evaluator instance
        evaluator = ModelEvaluator(model_path)
        
        # Get the corresponding label file path
        label_path = str(Path(image_path)).replace('images', 'labels').replace('.png', '.txt')
        
        # Load ground truth using evaluator
        ground_truth = evaluator.load_ground_truth(label_path)
        
        # Run prediction using evaluator's model
        results = evaluator.model.predict(
            source=image_path,
            conf=0.25,
            save=True,
            project="evaluation",
            name="results"
        )
        
        result = results[0]
        
        # Get class names mapping
        class_names = result.names
        
        # Count ground truth objects
        gt_counts = {}
        for gt in ground_truth:
            class_id = int(gt[0])
            class_name = class_names[class_id]
            gt_counts[class_name] = gt_counts.get(class_name, 0) + 1
        
        # Count predicted objects
        pred_counts = {}
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = class_names[class_id]
            confidence = float(box.conf[0])
            pred_counts[class_name] = pred_counts.get(class_name, 0) + 1
        
        # Prepare comparison table
        table_data = []
        all_classes = sorted(set(list(gt_counts.keys()) + list(pred_counts.keys())))
        
        for class_name in all_classes:
            gt_count = gt_counts.get(class_name, 0)
            pred_count = pred_counts.get(class_name, 0)
            
            # Calculate accuracy metrics
            if gt_count == 0 and pred_count == 0:
                accuracy = "N/A"
                status = "✓"
            elif gt_count == 0:
                accuracy = "False Positive"
                status = "❌"
            elif pred_count == 0:
                accuracy = "False Negative"
                status = "❌"
            else:
                accuracy = f"{min(gt_count, pred_count)/max(gt_count, pred_count):.2%}"
                status = "✓" if gt_count == pred_count else "❌"
            
            table_data.append([
                class_name,
                gt_count,
                pred_count,
                accuracy,
                status
            ])
        
        # Print results
        print("\nPrediction Evaluation:")
        print("-" * 50)
        print(f"Image: {os.path.basename(image_path)}")
        print(tabulate(table_data, 
                      headers=['Class', 'Ground Truth', 'Predicted', 'Accuracy', 'Match'],
                      tablefmt='grid'))
        
        # Calculate overall metrics
        total_gt = sum(gt_counts.values())
        total_pred = sum(pred_counts.values())
        print("\nOverall Metrics:")
        print(f"Total Ground Truth Objects: {total_gt}")
        print(f"Total Predicted Objects: {total_pred}")
        
        # Model confidence analysis for detected objects
        if result.boxes:
            confidences = [float(box.conf[0]) for box in result.boxes]
            print(f"\nConfidence Analysis:")
            print(f"Average Confidence: {np.mean(confidences):.2%}")
            print(f"Min Confidence: {min(confidences):.2%}")
            print(f"Max Confidence: {max(confidences):.2%}")
        
        print(f"\nAnnotated image saved in: {os.path.join('evaluation', 'results')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

def evaluate_batch(image_dir, model_path=None):
    """Evaluate multiple images in a directory"""
    if not os.path.exists(image_dir) or not os.path.isdir(image_dir):
        print(f"Error: {image_dir} does not exist or is not a directory")
        return
    
    # Initialize statistics
    total_stats = {
        'total_gt': 0,
        'total_pred': 0,
        'correct_pred': 0,
        'false_positives': 0,
        'false_negatives': 0
    }
    
    try:
        # Process all images in the directory
        for img_file in os.listdir(image_dir):
            if img_file.endswith(('.png', '.jpg')):
                img_path = os.path.join(image_dir, img_file)
                print(f"\nProcessing {img_file}...")
                evaluate_single_prediction(img_path, model_path)
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user. Saving partial results...")
    
    except Exception as e:
        print(f"\n\nAn error occurred: {str(e)}")
    
    finally:
        print("\nEvaluation Summary:")
        print("-" * 50)
        print(f"Total images processed: {len([f for f in os.listdir(os.path.join('evaluation', 'results')) if f.endswith('.png')])}")
        print(f"Results saved in: {os.path.join('evaluation', 'results')}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Evaluate model predictions against ground truth')
    parser.add_argument('path', type=str, help='Path to image file or directory')
    parser.add_argument('--model', type=str, help='Path to model weights', 
                      default='runs/detect/train/weights/best.pt')
    parser.add_argument('--batch', action='store_true', 
                      help='Process all images in directory')
    args = parser.parse_args()
    
    if args.batch:
        evaluate_batch(args.path, args.model)
    else:
        evaluate_single_prediction(args.path, args.model)