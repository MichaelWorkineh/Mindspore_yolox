from yolo_x import YOLO
from pathlib import Path
import cv2
import csv
from collections import Counter

print("Testing YOLO with webcam...")

class_counts = Counter()

# Override default names with provided categories
categories = [
    {"id": 0, "name": "objects"}, 
    {"id": 1, "name": "big bus"}, 
    {"id": 2, "name": "big truck"}, 
    {"id": 3, "name": "bus-l-"}, 
    {"id": 4, "name": "bus-s-"}, 
    {"id": 5, "name": "car"}, 
    {"id": 6, "name": "mid truck"}, 
    {"id": 7, "name": "small bus"}, 
    {"id": 8, "name": "small truck"}, 
    {"id": 9, "name": "truck-l-"}, 
    {"id": 10, "name": "truck-m-"}, 
    {"id": 11, "name": "truck-s-"}, 
    {"id": 12, "name": "truck-xl-"}
]
# Map id to name
custom_names = {cat['id']: cat['name'] for cat in categories}

try:
   
    path  = 'save/yolox/yolox-tiny_2-3_100.ckpt'

    # Load model
    model = YOLO(path, device="CPU")
    
    # Update model names map
    # Note: YOLOX output IDs might be 0-indexed corresponding to the order they were trained on.
    # If the training JSON had these IDs, the model output class '0' corresponds to 'objects', '5' to 'car', etc.
    # We update the model.names used by yolo_x.py results
    model.names = custom_names

    video = '4.mp4'
    # Test webcam access
    cap = cv2.VideoCapture(video)
    
    if not cap.isOpened():
        print("Error: Could not open video")
        exit()
    
    print("Video opened successfully. Starting detection...")
    
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps    = cap.get(cv2.CAP_PROP_FPS)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
    out = cv2.VideoWriter(
        'inference_output yolov12.mp4',
        fourcc,
        fps,
        (width, height)
    )

    while True:

        ret, frame = cap.read()
        if not ret:
            break
        
        # Run inference
        results = model(frame, conf=0.1)
        #results = dummy
        # Process results
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                
                # Convert to int
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                if conf > 0.1:
                    # Draw rectangle
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Print prediction to console
                    # Use custom_names directly or model.names which we updated
                    class_id = int(cls)
                    class_name = custom_names.get(class_id, f"Unknown-{class_id}")
                    
                    print(f"Detected: {class_name} with confidence: {conf:.2f}")
                    
                    # Update counts
                    class_counts[class_name] += 1
                    
                    # Add label to frame
                    label = f"{class_name} {conf:.2f}"
                    cv2.putText(frame, label, (x1, y1-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Write frame to output video
        out.write(frame)

        # Display the frame
        cv2.imshow('YOLO Detection', frame)
        
        # Break loop on 'q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except Exception as e:
    print(f"Error: {e}")
    print("Full error details:")

finally:
    # let ing the camera go and closing the window
    if 'cap' in locals():
        cap.release()
    if 'out' in locals():
        out.release()
    cv2.destroyAllWindows()
    
    # Write summary to CSV
    csv_file = 'detection_counts.csv'
    try:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Class Name', 'Count'])
            for class_name, count in class_counts.items():
                writer.writerow([class_name, count])
        print(f"Detection summary written to {csv_file}")
    except Exception as e:
        print(f"Error writing CSV: {e}")

    print("Cleanup completed")