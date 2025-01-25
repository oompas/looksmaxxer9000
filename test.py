import cv2
import mediapipe as mp

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize the webcam
cap = cv2.VideoCapture(1)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Failed to capture frame from camera. Check camera index.")
        break

    # Flip the image horizontally for a later selfie-view display
    image = cv2.flip(image, 1)
    
    # Convert the BGR image to RGB
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process the image and find facial landmarks
    results = face_mesh.process(rgb_image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get the coordinates of the eyes
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            # Calculate the average y-coordinate of the eyes
            avg_y = (left_eye.y + right_eye.y) / 2
            
            # Check if the person is looking at the camera
            if avg_y > 0.45:  # Adjust this threshold as needed
                cv2.putText(image, "Look at the camera!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Display the image
    cv2.imshow('Eye Tracking', image)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()
