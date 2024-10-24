import cv2
from ultralytics import solutions
from ultralytics import YOLO


model = YOLO("yolo11m.pt")
cap = cv2.VideoCapture("media/TestVideo.mp4")
assert cap.isOpened(), "Error reading video file"

# Get video properties: width, height, and frames per second (fps)
w, h, fps = (int(cap.get(x)) for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS))

# Define points for a line or region of interest in the video frame
region_points = [(20, 400), (1080, 404), (1080, 360), (20, 360)] # Line coordinates


counter = solutions.ObjectCounter(
    show=True,  # Display the image during processing
    region=region_points,  # Region of interest points
    model="yolo11m.pt",  # Ultralytics YOLO11 model file
    line_width=2,  # Thickness of the lines and bounding boxes
)


# Process video frames in a loop
while cap.isOpened():
    success, im0 = cap.read()
    print(im0.shape)
    if not success:
        print("Video frame is empty or video processing has been successfully completed.")
        break

    # Use the Object Counter to count objects in the frame and get the annotated image
    tracks = model.track(im0, persist=True, show=False, classes=[0])
    im0 = counter.count(im0, tracks)




