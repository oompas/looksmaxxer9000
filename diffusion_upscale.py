from super_image import EdsrModel, ImageLoader
from PIL import Image
import requests

# takes a lot of GPU VRAM

image = Image.open("./rage.png")
print(image.size)

model = EdsrModel.from_pretrained('eugenesiow/edsr-base', scale=2)
inputs = ImageLoader.load_image(image)
preds = model(inputs)

ImageLoader.save_image(preds, './scaled_2x.png')
ImageLoader.save_compare(inputs, preds, './scaled_2x_compare.png')