import cv2
import time
import os
from mtcnn import MTCNN

def create_new_faces(name):
    def capture_images(duration=6, max_images=10):
        parent_path = r"E:\DATA SCIENCE\programs\InFace\images"
        directory_path = os.path.join(parent_path, name)
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)

        # Open the default camera (usually the first camera, denoted by 0)
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Could not open camera.")
            return

        i = 0  # Counter for saved images
        start_time = time.time()  # Record the start time
        detector = MTCNN()

        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            faces = detector.detect_faces(rgb)

            for face in faces:
                x1, y1, width, height = face['box']
                cv2.rectangle(frame, (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 2)

            # Display the captured frame in a window
            cv2.imshow('Webcam', frame)

            elapsed_time = time.time() - start_time

            if elapsed_time <= duration and i < max_images:
                # Save the image to disk with incremented file name
                image_name = f'{name}_{i + 1}.jpg'
                storing_path = os.path.join(directory_path, image_name)
                cv2.imwrite(storing_path, frame)
                print(f"Image saved as {image_name}")
                i += 1
                time.sleep(0.5)  # Capture an image every 0.5 seconds (adjust if needed)

            # Break the loop if we captured the max number of images or the duration is over
            if i >= max_images or elapsed_time > duration:
                print("Image capture complete.")
                break

            # Check for user exit by pressing 'q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("Manual exit.")
                break

        # Release the webcam and close the window
        cap.release()
        cv2.destroyAllWindows()

    capture_images(duration=6, max_images=10)

# Example usage
# create_new_faces("student_name")  # Replace "student_name" with the actual name
