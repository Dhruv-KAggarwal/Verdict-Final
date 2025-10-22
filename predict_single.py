from ultralytics import YOLO
import argparse
import os

def predict_single_image(image_path):
    # Check if image exists
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        return
    
    try:
        # Load the model - using our best model from train3
        model = YOLO('runs/detect/train3/weights/best.pt')
        
        # Run prediction
        results = model.predict(
            source=image_path,
            conf=0.25,  # Confidence threshold
            save=True,  # Save results
            project="single_predictions",  # Project name
            name="results"  # Run name
        )
        
        # Get the first result (since we're only processing one image)
        result = results[0]
        
        print("\nDetections:")
        print("-----------")
        
        # If no detections found
        if len(result.boxes) == 0:
            print("No objects detected in the image.")
            return
        
        # Dictionary to count instances of each class
        class_counts = {}
        
        # Process each detection
        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = result.names[class_id]
            confidence = float(box.conf[0])
            
            # Update count
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
            
        # Print summary
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count} {'instance' if count == 1 else 'instances'}")
            
        print(f"\nPredicted image saved in: {os.path.join('single_predictions', 'results')}")
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect safety equipment in a single image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    args = parser.parse_args()
    
    predict_single_image(args.image_path)