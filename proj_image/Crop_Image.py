import cv2
import numpy as np
import matplotlib.pyplot as plt

image = cv2.imread('1.jpg')
# image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_numpy = np.array(image)
crop_image = image_numpy[400:700, :, :]
print(image_numpy.shape)
plt.imshow(crop_image)
plt.show()
cv2.imwrite('2.jpg', crop_image)
