import cv2
import mediapipe as mp
import math
import time
import numpy as np
import os
from datetime import datetime
from enhance_image import gray_world_balance, upscale, blur_background

screencap = None


runer = False
# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Start video capture from the second webcam
cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

def switch_blue_red_channels(image_path, output_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    if image is None:
        print("Error: Could not load image.")
        return
    if image.shape[2] != 4:
        print("Error: Image must have an alpha channel.")
        return
    image[:, :, [0, 2]] = image[:, :, [2, 0]]
    cv2.imwrite(output_path, image)
    print(f"Image saved to {output_path}")

# Switch channels for overlay images
switch_blue_red_channels("graphics/gandalf.png", "graphics/gandalf_switched.png")
switch_blue_red_channels("graphics/left_arrow.png", "graphics/left_arrow_switched.png")
switch_blue_red_channels("graphics/right_arrow.png", "graphics/right_arrow_switched.png")
switch_blue_red_channels("graphics/up_arrow.png", "graphics/up_arrow_switched.png")
switch_blue_red_channels("graphics/down_arrow.png", "graphics/down_arrow_switched.png")

# Load the overlay images
overlay_image = cv2.imread("graphics/gandalf_switched.png", cv2.IMREAD_UNCHANGED)
left_arrow = cv2.imread("graphics/left_arrow_switched.png", cv2.IMREAD_UNCHANGED)
right_arrow = cv2.imread("graphics/right_arrow_switched.png", cv2.IMREAD_UNCHANGED)
up_arrow = cv2.imread("graphics/up_arrow_switched.png", cv2.IMREAD_UNCHANGED)
down_arrow = cv2.imread("graphics/down_arrow_switched.png", cv2.IMREAD_UNCHANGED)

if overlay_image is None or left_arrow is None or right_arrow is None or up_arrow is None or down_arrow is None:
    print("Error: Could not load overlay images.")
    exit()

if overlay_image.shape[2] != 4:
    print("Error: graphics/gandalf_switched.png must have an alpha channel.")
    exit()

def calculate_average_rgb(image, center_x, center_y, radius):
    x_min = max(0, center_x - radius)
    x_max = min(image.shape[1], center_x + radius)
    y_min = max(0, center_y - radius)
    y_max = min(image.shape[0], center_y + radius)
    roi = image[y_min:y_max, x_min:x_max]
    return roi.mean(axis=(0, 1))

def angle_to_color(angle):
    green_intensity = max(0, min(255, 255 - abs(angle)*25))
    red_intensity = max(0, min(255, abs(angle)*25))
    return (red_intensity, green_intensity, 0)

green_rectangle_start_time = None
arrows_not_present_start_time = None
good_angle_start_time = None
conditions_met_duration = 2
countdown_started = False
countdown_time = 3
countdown_start_time = 0
countdown_done = False
green_rectangle_time = None
last_light_check_time = time.time()
light_check_interval = 10
dot_scale = 0.1

flash_image = np.ones((720, 1280, 3), dtype=np.uint8) * 255
flash_alpha = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    frame = cv2.flip(frame, 1)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_height, frame_width = frame.shape[:2]

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
            x_min, y_min = min(x_min, x), min(y_min, y)
            x_max, y_max = max(x_max, x), max(y_max, y)
            mesh_center_x += x
            mesh_center_y += y

            # Draw green dots for each landmark
            cv2.circle(annotated_image, (x, y), 1, (0, 255, 0), -1)

        # Calculate the average center of the mesh
        num_landmarks = len(face_landmarks.landmark)
        mesh_center_x //= num_landmarks
        mesh_center_y //= num_landmarks
        #print(f"Average Mesh Center ({mesh_center_x}, {mesh_center_y})")

        # Top 1/3 of face lighting adjustments
        y_min = min(int(landmark.y * frame_height) for landmark in face_landmarks.landmark)
        y_max = max(int(landmark.y * frame_height) for landmark in face_landmarks.landmark)
        upper_face_threshold = y_min + (y_max - y_min) / 3

        # Calculate average lighting, distance, and angle of landmark points to mesh center
        landmark_data = []
        for landmark in face_landmarks.landmark:
            x = int(landmark.x * frame_width)
            y = int(landmark.y * frame_height)

            avg_rgb = calculate_average_rgb(frame, x, y, radius=10)
            total_avg_rgb = (avg_rgb[0] + avg_rgb[1] + avg_rgb[2])/3
            distance = math.sqrt((x - mesh_center_x)**2 + (y - mesh_center_y)**2)
            angle_radians = math.atan2(y - mesh_center_y, x - mesh_center_x)
            #angle_degrees = math.degrees(math.atan2(y - mesh_center_y, x - mesh_center_x))

            # Top 1/3 of face lighting adjustments
            if y <= upper_face_threshold:
                distance += 20
            
            landmark_data.append((x, y, distance, angle_radians, total_avg_rgb))

        # Create vectors weighted proportionally to its distance and lighting intensity
        weighted_x_sum = 0
        weighted_y_sum = 0
        #print("Landmark data (x, y, distance, angle, total_light):")
        for x, y, distance, angle_radians, total_avg_rgb in landmark_data:
            #print(f"Landmark ({x}, {y}): Distance = {distance:.2f}, Angle = {angle:.1f}°, Total Avg RGB = {total_avg_rgb:.0f}")
            weighted_intensity = distance * total_avg_rgb
            vector_x = weighted_intensity * math.cos(angle_radians)
            vector_y = weighted_intensity * math.sin(angle_radians)

            weighted_x_sum += vector_x
            weighted_y_sum += vector_y

        overall_angle = math.degrees(math.atan2(weighted_y_sum, weighted_x_sum))
        #sundial area
        #print(f"Overall Lighting Angle: {overall_angle:.2f}°")

        # Draw the lighting intensity angle
        try:
            arrow_length = 50
            arrow_end_x = int(mesh_center_x + arrow_length * math.cos(math.radians(overall_angle)))
            arrow_end_y = int(mesh_center_y + arrow_length * math.sin(math.radians(overall_angle)))
        except:
            pass
        cv2.arrowedLine(
            annotated_image,
            (mesh_center_x, mesh_center_y),
            (arrow_end_x, arrow_end_y),
            (255, 255, 0),
            2,
            tipLength=0.2
        )

        # Calculate face bounding box size and percentage of screen
        face_width = x_max - x_min
        face_height = y_max - y_min
        face_area = face_width * face_height
        frame_area = frame_width * frame_height
        face_percentage = (face_area / frame_area) * 100

        face_center_y = (y_min + y_max) // 2
        vertical_center_start = int(frame_height * 0.45)
        vertical_center_end = int(frame_height * 0.52)

        rectangle_color = (0, 255, 0) if (15 <= face_percentage <= 24 and
                                          vertical_center_start <= face_center_y <= vertical_center_end) else (255, 0, 0)
        cv2.rectangle(annotated_image, (x_min, y_min), (x_max, y_max), rectangle_color, 2)

        current_time = time.time()

        if rectangle_color == (0, 255, 0):
            if green_rectangle_start_time is None:
                green_rectangle_start_time = current_time
        else:
            green_rectangle_start_time = None

        # Check face position and size
        if face_center_y < vertical_center_start:
            dot_x = int(mesh_center_x)
            dot_y = y_max + int(1.5 * dot_scale * face_height)
            arrow_image = down_arrow
        elif face_center_y > vertical_center_end:
            dot_x = int(mesh_center_x)
            dot_y = y_min - int(3 * dot_scale * face_height)
            arrow_image = up_arrow


        if face_percentage < 15:
            cv2.putText(annotated_image, "Approach camera", (frame_width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        elif face_percentage > 24:
            cv2.putText(annotated_image, "Move away from camera", (frame_width - 250, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        middle_15_percent_start = frame_width * 0.425
        middle_15_percent_end = frame_width * 0.575

        if not (middle_15_percent_start <= mesh_center_x <= middle_15_percent_end):
            if mesh_center_x < middle_15_percent_start:
                dot_x = x_max + int(3 * dot_scale * face_width)
                dot_y = int(mesh_center_y - 0.1 * face_height)
                arrow_image = left_arrow
            elif mesh_center_x > middle_15_percent_end:
                dot_x = x_min - int(3 * dot_scale * face_width)
                dot_y = int(mesh_center_y - 0.1 * face_height)
                arrow_image = right_arrow

        if dot_x is not None and dot_y is not None:
            arrow_height, arrow_width = arrow_image.shape[:2]
            arrow_new_height = int(0.3 * face_height)
            aspect_ratio = arrow_width / arrow_height
            arrow_new_width = int(arrow_new_height * aspect_ratio)
            arrow_resized = cv2.resize(arrow_image, (arrow_new_width, arrow_new_height), interpolation=cv2.INTER_AREA)

            arrow_x_start = max(0, dot_x - arrow_new_width // 2)
            arrow_y_start = max(0, dot_y - arrow_new_height // 2)
            arrow_x_end = min(frame_width, arrow_x_start + arrow_new_width)
            arrow_y_end = min(frame_height, arrow_y_start + arrow_new_height)

            arrow_resized = arrow_resized[:arrow_y_end - arrow_y_start, :arrow_x_end - arrow_x_start]
            arrow_alpha = arrow_resized[:, :, 3] / 255.0

            for c in range(3):
                raw_feed_image[
                    arrow_y_start:arrow_y_start + arrow_resized.shape[0],
                    arrow_x_start:arrow_x_start + arrow_resized.shape[1],
                    c
                ] = (
                    arrow_alpha * arrow_resized[:, :, c] +
                    (1 - arrow_alpha) * raw_feed_image[
                        arrow_y_start:arrow_y_start + arrow_resized.shape[0],
                        arrow_x_start:arrow_x_start + arrow_resized.shape[1],
                        c
                    ]
                )

        left_eye = face_landmarks.landmark[33]
        right_eye = face_landmarks.landmark[263]
        left_eye_x, left_eye_y = int(left_eye.x * frame_width), int(left_eye.y * frame_height)
        right_eye_x, right_eye_y = int(right_eye.x * frame_width), int(right_eye.y * frame_height)

        eye_slope = (right_eye_y - left_eye_y) / (right_eye_x - left_eye_x + 1e-5)
        head_angle = math.degrees(math.atan(eye_slope))

        angle_text_color = angle_to_color(head_angle)
        head_angle_text = f"Angle: {head_angle:.2f}"
        if -2.5 <= head_angle <= 2.5:
            head_angle_text += " Good"

        cv2.putText(
            annotated_image,
            head_angle_text,
            (x_min, y_min - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            angle_text_color,
            2,
        )

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            
            # Get the coordinates of the eyes
            left_eye = face_landmarks.landmark[33]
            right_eye = face_landmarks.landmark[263]
            
            # Calculate the average y-coordinate of the eyes
            avg_y = (left_eye.y + right_eye.y) / 2
            
            # Check if the person is looking at the camera
            if avg_y > 0.45:  # Adjust this threshold as needed
                cv2.putText(annotated_image, "Look at the camera!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)


        if current_time - last_light_check_time >= light_check_interval:
            last_light_check_time = current_time
            # print("Average RGB light levels (out of 255) for each landmark:")
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * frame_width), int(landmark.y * frame_height)
                avg_rgb = calculate_average_rgb(frame, x, y, radius=10)
                # print(f"Landmark ({x}, {y}): Total={(avg_rgb[0] + avg_rgb[1] + avg_rgb[2])/3:.0f}, R={avg_rgb[2]:.0f}, G={avg_rgb[1]:.0f}, B={avg_rgb[0]:.0f}")
        # Calculate the position to center the overlay on the face
        x_start = x_min + (x_max - x_min) // 4
        y_start = y_min + (y_max - y_min) // 4
    if flash_image.shape[:2] != raw_feed_image.shape[:2]:
        flash_image = cv2.resize(flash_image, (raw_feed_image.shape[1], raw_feed_image.shape[0]))

    if flash_alpha > 0:
        for c in range(3):
            raw_feed_image[:, :, c] = (
                flash_alpha * flash_image[:, :, c] + (1 - flash_alpha) * raw_feed_image[:, :, c]
            )
            annotated_image[:, :, c] = (
                flash_alpha * flash_image[:, :, c] + (1 - flash_alpha) * annotated_image[:, :, c]
            )
        flash_alpha -= 0.1
    current_time = time.time()
    if dot_x is None and dot_y is None:
        if arrows_not_present_start_time is None:
            arrows_not_present_start_time = current_time
    else:
        arrows_not_present_start_time = None

    if "Good" in head_angle_text:
        if good_angle_start_time is None:
            good_angle_start_time = current_time
    else:
        good_angle_start_time = None

    all_conditions_met = (green_rectangle_start_time is not None and
                          arrows_not_present_start_time is not None and
                          good_angle_start_time is not None)

    if cv2.waitKey(1) & 0xFF == ord("y"):
        runer = True

    if all_conditions_met and not runer:
        elapsed_time = min(current_time - green_rectangle_start_time,
                           current_time - arrows_not_present_start_time,
                           current_time - good_angle_start_time)
        

        if elapsed_time >= conditions_met_duration and not countdown_started and not countdown_done:
            countdown_started = True
            countdown_start_time = current_time
    elif runer:
        elapsed_time = current_time
        
        if elapsed_time >= conditions_met_duration and not countdown_started and not countdown_done:
            countdown_started = True
            countdown_start_time = current_time

    if countdown_started and not countdown_done:
        elapsed_time = current_time - countdown_start_time
        remaining_time = max(0, countdown_time - int(elapsed_time))

        cv2.putText(
            annotated_image,
            str(remaining_time),
            (frame_width // 2 - 30, frame_height // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (0, 255, 255),
            3,
        )

        if remaining_time == 0:
            countdown_done = True
            countdown_started = False
            screencap = frame.copy()
            flash_alpha = 1.0
                        # Create 'pictures' folder if it doesn't exist
            if not os.path.exists('pictures'):
                os.makedirs('pictures')
            
            # Generate unique filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"pictures/screencap_{timestamp}.png"
            screencap = upscale(screencap, 2)
            
            # Save the screencap
            cv2.imwrite(filename, screencap)
            print(f"Screencap saved as {filename}")

    cv2.imshow("Annotated Feed", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))
    cv2.imshow("Raw Feed", cv2.cvtColor(raw_feed_image, cv2.COLOR_RGB2BGR))

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

