import cv2

img = cv2.imread('leaf.jpg')

img = cv2.resize(img, None, fx=0.5, fy=0.5)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
gray = clahe.apply(gray)

blur = cv2.GaussianBlur(gray, (5, 5), 0)

edges_low = cv2.Canny(blur, 20, 60)
edges_mid = cv2.Canny(blur, 50, 120)
edges_high = cv2.Canny(blur, 100, 200)

cv2.imshow('Original', img)
cv2.imshow('Veins low (20, 60)', edges_low)
cv2.imshow('Veins mid (50, 120)', edges_mid)
cv2.imshow('Veins high (100, 200)', edges_high)

cv2.waitKey(0)
cv2.destroyAllWindows()