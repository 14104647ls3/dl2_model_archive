import yolov5
import cv2

# Load the YOLOv5 model
model = yolov5.load('best_s.pt')

# Open the video file
cap = cv2.VideoCapture('InferenceVideoRaw.mp4')

# Output video file
output_file = 'output_video_yolo_s.mp4'

# Get the video's frame width and height
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Define the codec and create VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_file, fourcc, 20.0, (frame_width, frame_height))

# Process each frame of the video
while True:
    ret, frame = cap.read()
    
    # Check if there are no more frames to read
    if not ret:
        break

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Perform object detection
    results = model(frame_rgb)
    
    # Draw bounding boxes on the frame
    for pred in results.pred[0]:
        xmin, ymin, xmax, ymax, conf, cls = map(int, pred[:6])
        class_name = model.names[int(cls)]
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        cv2.putText(frame, f'{class_name} {conf:.2f}', (xmin, ymin - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Write the frame with bounding boxes to the output video
    out.write(frame)

# Release video objects
cap.release()
out.release()
cv2.destroyAllWindows()
