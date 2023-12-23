import cv2


#cap = cv2.VideoCapture("Untitled video - Made with Clipchamp.mp4") 
cap = cv2.VideoCapture("737b07de-8296-4495-8e62-e136d576176f.mp4") 


desired_fps = 30


current_fps = int(cap.get(cv2.CAP_PROP_FPS))


if current_fps != desired_fps:
    
    cap.set(cv2.CAP_PROP_FPS, desired_fps)
    print(f"FPS set to {desired_fps}")

while True:
    ret, frame = cap.read()

    if not ret:
        break

    
    cv2.imshow('Video', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()