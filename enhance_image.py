import cv2
import numpy as np

# upscaling
def upscale(img, scale_factor=2):
    height, width = img.shape[:2]
    new_dimensions = (width * scale_factor, height * scale_factor)
    res_img = cv2.resize(img, dsize=new_dimensions, interpolation=cv2.INTER_CUBIC)
    
    return res_img

# adjust colour balance
def gray_world_balance(img):
    b, g, r = cv2.split(img)
    
    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)
    
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r
    
    b = cv2.convertScaleAbs(b, alpha=scale_b)
    g = cv2.convertScaleAbs(g, alpha=scale_g)
    r = cv2.convertScaleAbs(r, alpha=scale_r)
    
    return cv2.merge([b, g, r])


def blur_background(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Load Haar cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Create a blank mask
    mask = np.zeros_like(image)

    for (x, y, w, h) in faces:
        # Face rectangle (to keep face unblurred)
        cv2.rectangle(mask, (x, y), (x + w, y + h), (255, 255, 255), -1)

        # Circle for hair
        center = (x + w // 2, y)  # Center above the face
        radius = int(h * 0.6)     # Radius based on face height
        cv2.circle(mask, center, radius, (255, 255, 255), -1)

        # Trapezoid for shoulders
        top_left = (x - int(w * 0.5), y + h)
        top_right = (x + w + int(w * 0.5), y + h)
        bottom_left = (x - int(w * 1.0), y + int(h * 2))
        bottom_right = (x + w + int(w * 1.0), y + int(h * 2))
        trapezoid = np.array([top_left, top_right, bottom_right, bottom_left], dtype=np.int32)
        cv2.fillPoly(mask, [trapezoid], (255, 255, 255))

        # Triangle from top of hair to chin
        top_of_hair = (x + w // 2, y - int(h * 0.2))  # Adjust top point for hair extension
        left_chin = (x, y + h)  # Left edge of the chin
        right_chin = (x + w, y + h)  # Right edge of the chin
        triangle = np.array([top_of_hair, left_chin, right_chin], dtype=np.int32)
        cv2.fillPoly(mask, [triangle], (255, 255, 255))

    # Create the blurred background
    background_mask = cv2.bitwise_not(mask)
    blurred = cv2.GaussianBlur(image, (71, 71), 0)
    foreground = cv2.bitwise_and(image, mask)
    background = cv2.bitwise_and(blurred, background_mask)
    result = cv2.add(foreground, background)

    return result

# to test
# image_path = './tkd.jpg'
# img = cv2.imread(image_path)
# print(img.shape)
# res_img = blur_background(gray_world_balance(upscale(img)))
# cv2.imwrite('enhanced.jpg', res_img)
# print(res_img.shape)