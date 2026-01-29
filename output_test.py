from yolo_x import YOLO

# Load model
model = YOLO("save/yolox/yolox-tiny_2-3_100.ckpt", device="CPU")

# Run inference (returns list of Results objects)
results = model("testImage/adit_mp4-19_jpg.rf.a7a2617832c11b8ebd0a1b3e833807c9.jpg", conf=0.1)

for result in results:
    # Access boxes (mimics result.boxes.xyxy)
    print(result.boxes.xyxy) 
    # Plot
    result.save("output_v12.jpg")