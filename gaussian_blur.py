import cv2
import numpy as np
import os

img = cv2.imread('Leaf.jpg')

img = cv2.resize(img, (300, 300))

blur_small = cv2.GaussianBlur(img, (5, 5), 0)
blur_medium = cv2.GaussianBlur(img, (15, 15), 0)
blur_large = cv2.GaussianBlur(img, (45, 45), 0)

def label(image, text):
    out = image.copy()
    cv2.putText(out, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.8, (0, 255, 0), 2)
    return out

row = np.hstack([
    label(img, "Original"),
    label(blur_small, "5x5"),
    label(blur_medium, "15x15"),
    label(blur_large, "45x45"),
])

os.makedirs("docs", exist_ok=True)
cv2.imwrite("docs/leaf_blur_result.png", row)

cv2.imshow("Leaf Gaussian blur", row)
cv2.waitKey(0)
cv2.destroyAllWindows()