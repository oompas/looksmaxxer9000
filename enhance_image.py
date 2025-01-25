import cv2
import numpy as np

image_path = './rabab.jpeg'

img = cv2.imread(image_path)
print(img.shape)

# upscaling
scale_factor = 4
height, width = img.shape[:2]
new_dimensions = (width * scale_factor, height * scale_factor)
res_img = cv2.resize(img, dsize=new_dimensions, interpolation=cv2.INTER_CUBIC)

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

res_img = gray_world_balance(res_img)
cv2.imwrite('enhanced.jpg', res_img)
print(res_img.shape)