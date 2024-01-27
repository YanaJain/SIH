import cv2

# Open a video capture object (0 for the default camera)
cap = cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4")

# Set the desired FPS (change this value as needed)
desired_fps = 30

# Get the current frames per second of the video capture device
current_fps = int(cap.get(cv2.CAP_PROP_FPS))

# Check if the camera supports the desired FPS
if current_fps != desired_fps:
    # Set the desired FPS
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    print(f"FPS set to {desired_fps}")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    # Your processing on the frame goes here
    # For example, you can display the frame
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()