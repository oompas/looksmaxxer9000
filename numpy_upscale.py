import cv2

img = cv2.imread('./image2.jpg')

scale_factor = 4
height, width = img.shape[:2]
new_dimensions = (width * scale_factor, height * scale_factor)

res = cv2.resize(img, dsize=new_dimensions, interpolation=cv2.INTER_CUBIC)

cv2.imwrite("./scaled_rage.png", res)
print(res.shape)