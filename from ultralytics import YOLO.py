from ultralytics import YOLO

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

source = "VID1.mp4"
# Run batched inference on a video, with vid_stride to skip frames
results = model(source, stream=True, vid_stride=600)  # process every 60th frame

# Process results generator
for result in results:
    boxes = result.boxes  
    masks = result.masks  
    keypoints = result.keypoints  
    probs = result.probs  
    obb = result.obb  
    result.show()  # display to screen
    result.save(filename="result.jpg")  # save to disk
