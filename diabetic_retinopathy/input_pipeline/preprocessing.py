import torch
import cv2
import PIL.Image as Image
import os
import numpy as np
from matplotlib import pyplot as plt
import torchvision.transforms as tf
from torchvision.utils import save_image


h, w = 712, 1072
path = os.path.join(r"C:\Users\guodiandian\Downloads\idrid\IDRID_dataset\images\train\IDRiD_001.jpg")
output = os.path.join(r"C:\Users\guodiandian\Downloads\newimg.jpg")


img = Image.open(path)
img = img.resize([w, h])
img2 = img.transpose(Image.FLIP_LEFT_RIGHT)
img = np.array(img)
img2 = np.array(img2)
img_red = img[:, :, 0]
img_red2 = img2[:, :, 0]
res = 1000
res2 = 1000

for i in range(img_red.shape[0]):
    for j in range(img_red.shape[1]):
        if img_red[i][j] > 20 and j < res:
            res = j
            break

for i in range(img_red2.shape[0]):
    for j in range(img_red2.shape[1]):
        if img_red2[i][j] > 20 and j < res2:
            res2 = j
            break

a, b = res, w-res2
img = img[:, a:b, :]
# plt.imshow(img)
# plt.show()
img = np.transpose(img, (2, 0, 1))
img = torch.tensor(img)
img = tf.Resize([512, 512])(img)
img = img/255.
save_image(img, output)





