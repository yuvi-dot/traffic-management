from ultralytics import YOLO
import csv

# Define the length of the road section in view (in kilometers or meters)
road_length_km = 0.5  # Example: 0.5 kilometers

# Load a model
model = YOLO("yolov8n.pt")  # pretrained YOLOv8n model

# List of video sources to process
video_sources = ["VID1.mp4", "VID2.mp4","VID3.mp4","VID4.mp4"]

# Define class IDs for vehicles (car, bus, truck) - typically COCO dataset IDs
vehicle_classes = {2: "car", 5: "bus", 7: "truck"}

# Loop over each video in the list
for video in video_sources:
    # Create a CSV file for each video
    csv_filename = f"traffic_density_{video}.csv"  # Unique filename per video

    # Open a CSV file to write the results for this video
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write the header row
        writer.writerow(["video", "frame", "car_count", "bus_count", "truck_count", "total_vehicles", "traffic_density (vehicles/km)"])

        # Process the video
        results = model(video, stream=True, vid_stride=60)  # process every 60th frame

        # Process results generator
        for frame_idx, result in enumerate(results):
            boxes = result.boxes  # Bounding box outputs

            # Initialize vehicle counts for the frame
            car_count = 0
            bus_count = 0
            truck_count = 0
            
            if boxes is not None:
                for box in boxes:
                    cls = int(box.cls[0])  # Get class label for each detected object
                    if cls in vehicle_classes:
                        if cls == 2:  # car
                            car_count += 1
                        elif cls == 5:  # bus
                            bus_count += 1
                        elif cls == 7:  # truck
                            truck_count += 1
            
            # Calculate the total number of vehicles in the frame
            total_vehicles = car_count + bus_count + truck_count
            
            # Calculate traffic density (vehicles per kilometer)
            traffic_density = total_vehicles / road_length_km
            
            # Write data to CSV: [video, frame, car_count, bus_count, truck_count, total_vehicles, traffic_density]
            writer.writerow([video, frame_idx + 1, car_count, bus_count, truck_count, total_vehicles, traffic_density])

    print(f"Traffic density saved to '{csv_filename}' for video {video}.")
