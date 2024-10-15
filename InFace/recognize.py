import cv2
def recognize_face():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    ret, frame = cap.read()
    if ret:
        # Display the captured frame
        cv2.imshow("Captured Photo", frame)
        # Save the captured image to a file
        cv2.imwrite("captured_photo.jpg", frame)
        # Wait for a key press and close the window
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: Could not read frame.")
    return frame
