#https://stackoverflow.com/questions/24385714/detect-text-region-in-image-using-opencv
import cv2
import numpy as np
import pytesseract
from itertools import product

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Load image, convert to HSV format, define lower/upper ranges, and perform
# color segmentation to create a binary mask

image = cv2.imread(r'C:\Users\parvesh.dhawan\Desktop\safaltek_p1\test_images\00a05e10-d923-4c3f-bae6-09c753440b48.jpg')
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

for i, j, k in product(range(179), range(255), range(255)):
    lower = np.array([i,j,k])
    upper = np.array([179, 255, 255])
    mask = cv2.inRange(hsv, lower, upper)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5,3))
    dilate = cv2.dilate(mask, kernel, iterations=5)
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    for c in cnts:
        x,y,w,h = cv2.boundingRect(c)
        ar = w / float(h)
        if ar < 5:
            cv2.drawContours(dilate, [c], -1, (0,0,0), -1)

    # Bitwise dilated image with mask, invert, then OCR
    result = 255 - cv2.bitwise_and(dilate, mask)
    data = pytesseract.image_to_string(result, lang='eng',config='--psm 6')
    if len(data)!=0:
        print(data)
        print('------------------------------------------')
        break
    else:
        continue

print('---')

# cv2.imshow('mask', mask)
# cv2.imshow('dilate', dilate)
# cv2.imshow('result', result)
# cv2.waitKey(0)
