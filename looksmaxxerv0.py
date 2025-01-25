import cv2
import mediapipe as mp
import math
import time
import numpy as np

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5
)

# Start video capture from the second webcam
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def switch_blue_red_channels(image_path, output_path):
    # Load the image
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not load image.")
        return

    # Ensure the image has an alpha channel
    if image.shape[2] != 4:
        print("Error: Image must have an alpha channel.")
        return

    # Switch the blue and red channels
    image[:, :, [0, 2]] = image[:, :, [2, 0]]

    # Save the modified image
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

# Example usage
switch_blue_red_channels("gandalf.png", "gandalf_switched.png")

# Load the overlay image
overlay_image = cv2.imread("gandalf_switched.png", cv2.IMREAD_UNCHANGED)
if overlay_image is None:
    print("Error: Could not load gandalf_switched.png.")
    exit()

# Ensure overlay image has an alpha channel
if overlay_image.shape[2] != 4:
    print("Error: gandalf_switched.png must have an alpha channel.")
    exit()

def calculate_average_rgb(image, center_x, center_y, radius):
    x_min = max(0, center_x - radius)
    x_max = min(image.shape[1], center_x + radius)
    y_min = max(0, center_y - radius)
    y_max = min(image.shape[0], center_y + radius)

    roi = image[y_min:y_max, x_min:x_max]
    average_color = roi.mean(axis=(0, 1))  # Calculate mean across height and width
    return average_color

# Function to calculate the color based on head angle
def angle_to_color(angle):
    green_intensity = max(0, min(255, 255 - abs(angle)*25))
    red_intensity = max(0, min(255, abs(angle)*25))
    return (red_intensity, green_intensity, 0)

countdown_started = False
countdown_time = 3
countdown_start_time = 0
countdown_done = False  # Flag to ensure countdown runs only once

# Time delay before starting countdown
green_rectangle_time = None

# Initialize timer for light level checks
last_light_check_time = time.time()
light_check_interval = 10  # Check every 10 seconds

# Scaling factor for dot placement
dot_scale = 0.1

while True:
    # Capture frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break
    frame = cv2.flip(frame, 1)

    # Convert the frame to RGB (Mediapipe requires RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]

    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(frame_rgb)

    annotated_image = frame_rgb.copy()
    raw_feed_image = frame_rgb.copy()

    dot_x, dot_y = None, None
    head_angle_text = ""

    if results.multi_face_landmarks:
        # Process the first detected face
        face_landmarks = results.multi_face_landmarks[0]

        # Initialize bounding box and calculate mesh center
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        mesh_center_x, mesh_center_y = 0, 0

        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x)
            y_max = max(y_max, y)
            mesh_center_x += x
            mesh_center_y += y

            # Draw green dots for each landmark
            cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

        # Calculate the average center of the mesh
        num_landmarks = len(face_landmarks.landmark)
        mesh_center_x //= num_landmarks
        mesh_center_y //= num_landmarks

        # Calculate face bounding box size and percentage of screen
        face_width = x_max - x_min
        face_height = y_max - y_min
        face_area = face_width * face_height
        frame_area = frame_width * frame_height
        face_percentage = (face_area / frame_area) * 100

        # Determine rectangle color based on face percentage
        rectangle_color = (0, 255, 0) if 20 <= face_percentage <= 35 else (255, 0, 0)

        # Draw the bounding box
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), rectangle_color, 2)

        # Check if the rectangle turned green and start a timer
        if rectangle_color == (0, 255, 0) and not countdown_started and green_rectangle_time is None:
            green_rectangle_time = time.time()  # Record the time when the rectangle turns green

        # Start countdown after 2 seconds delay
        if green_rectangle_time and not countdown_started:
            elapsed_time = time.time() - green_rectangle_time
            if elapsed_time >= 2:  # 2 seconds delay
                countdown_started = True
                countdown_start_time = time.time()  # Start countdown

        # If countdown has started, display the countdown
        if countdown_started and not countdown_done:
            elapsed_time = time.time() - countdown_start_time
            remaining_time = max(0, countdown_time - int(elapsed_time))

            # Draw the countdown in the center of the frame
            cv2.putText(
                annotated_image,
                str(remaining_time),
                (frame_width // 2 - 30, frame_height // 2),
                cv2.FONT_HERSHEY_SIMPLEX,
                2,
                (0, 255, 255),  # Yellow color
                3,
            )

            # Stop countdown when it reaches 0
            if remaining_time == 0:
                countdown_done = True
                countdown_started = False

        # Middle 15% of the frame width
        middle_15_percent_start = frame_width * 0.425
        middle_15_percent_end = frame_width * 0.575

        # Check if the mesh center's x-axis is within the middle 15% of the frame
        if not (middle_15_percent_start <= mesh_center_x <= middle_15_percent_end):
            # Face is off-center horizontally
            if mesh_center_x < middle_15_percent_start:  # Face is on the left
                dot_x = x_max + int(dot_scale * face_width)
                dot_y = mesh_center_y
            elif mesh_center_x > middle_15_percent_end:  # Face is on the right
                dot_x = x_min - int(dot_scale * face_width)
                dot_y = mesh_center_y

        # Calculate head angle based on key facial landmarks
        left_eye = face_landmarks.landmark[33]  # Left eye outer corner
        right_eye = face_landmarks.landmark[263]  # Right eye outer corner

        # Map landmark coordinates to screen dimensions
        left_eye_x = int(left_eye.x * frame_width)
        left_eye_y = int(left_eye.y * frame_height)
        right_eye_x = int(right_eye.x * frame_width)
        right_eye_y = int(right_eye.y * frame_height)

        # Calculate angle of the head using eyes as reference
        eye_slope = (right_eye_y - left_eye_y) / (right_eye_x - left_eye_x + 1e-5)  # Prevent division by zero
        head_angle = math.degrees(math.atan(eye_slope))

        # Get the color for the angle text based on the head angle
        angle_text_color = angle_to_color(head_angle)

        # Display head angle above the bounding box with dynamic color
        head_angle_text = f"Angle: {head_angle:.2f}"
        cv2.putText(
            annotated_image,
            head_angle_text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            angle_text_color,
            2,
        )

        # Check the average RGB light levels every 10 seconds
        current_time = time.time()
        if current_time - last_light_check_time >= light_check_interval:
            last_light_check_time = current_time
            print("Average RGB light levels (out of 255) for each landmark:")
            for landmark in face_landmarks.landmark:
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                avg_rgb = calculate_average_rgb(frame, x, y, radius=10)
                print(f"Landmark ({x}, {y}): Total={(avg_rgb[0] + avg_rgb[1] + avg_rgb[2])/3:.0f}, R={avg_rgb[2]:.0f}, G={avg_rgb[1]:.0f}, B={avg_rgb[0]:.0f}")

        # Overlay the Gandalf image on the raw feed
        overlay_height, overlay_width = overlay_image.shape[:2]
        center_x = (x_min + x_max) // 2
        center_y = (y_min + y_max) // 2

        x_start = max(0, center_x - overlay_width // 2)
        y_start = max(0, center_y - overlay_height // 2)
        x_end = min(frame_width, x_start + overlay_width)
        y_end = min(frame_height, y_start + overlay_height)

        overlay_resized = cv2.resize(overlay_image, (x_end - x_start, y_end - y_start))

        # Combine the overlay image with the raw feed
        alpha = overlay_resized[:, :, 3] / 255.0  # Extract the alpha channel
        for c in range(3):  # Blend R, G, B channels
            raw_feed_image[y_start:y_end, x_start:x_end, c] = (
                alpha * overlay_resized[:, :, c] + (1 - alpha) * raw_feed_image[y_start:y_end, x_start:x_end, c]
            )

    # Draw the guiding dot if needed (only on annotated image)
    if dot_x is not None and dot_y is not None:
        cv2.circle(annotated_image, (dot_x, dot_y), 10, (255, 0, 0), -1)

    # Show the frames
    cv2.imshow("Annotated Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Raw Feed", cv2.cvtColor(raw_feed_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
