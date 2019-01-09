import cv2
import numpy as np
cap = cv2.VideoCapture(0)

#	#For Full Screen uncomment below

#cv2.namedWindow('title', cv2.WND_PROP_FULLSCREEN)
#cv2.setWindowProperty('title', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while True:
    ret,frame = cap.read()
    cv2.imshow('Original',frame)
    # 1. Blur
    blur = cv2.medianBlur(frame,15)
    
    # 2. To Gray
    gray = cv2.cvtColor(blur,cv2.COLOR_BGR2GRAY)
    
    # 3.) Threshold
    ret, thresh = cv2.threshold(gray,160,255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # 4.) noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    
    # 5.) sure background area
    bg = cv2.dilate(opening,kernel,iterations=3)
    
    # 6.) Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, fg = cv2.threshold(dist_transform,0.3*dist_transform.max(),255,0)
    
    # 7.) Sure Forground region
    final_fg = np.uint8(fg)
    
    # 8.) unknown region
    unknown = cv2.subtract(bg, final_fg)
    
    # 9.) Label MArker for sure Foreground
    # Marker labelling
    ret, markers = cv2.connectedComponents(final_fg)
    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1
    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0
    
    #10.) applying watershed algorithm
    water_markers = cv2.watershed(frame,markers)
    
    # 11.) Finding Contours
    image, contours, hierarchy = cv2.findContours(water_markers.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    # For every entry in contours
    for i in range(len(contours)):

        # last column in the array is -1 if an external contour (no contours inside of it)
        if hierarchy[0][i][3] == -1:

            # We can now draw the external contours from the list of contours
            cv2.drawContours(frame, contours, i, (255, 0, 0), 10)
        

    #Displaying IMage
#     cv2.imshow('1.) Blur',blur)
#     cv2.imshow('2.) Gray',gray)
#     cv2.imshow('3.) Threshold',thresh)
#     cv2.imshow('4.) Opening-Noise Removal',opening)
#     cv2.imshow('5.) BG',bg)
#     cv2.imshow('6.) FG',fg)
#     cv2.imshow('7.) Final FG',final_fg)
#     cv2.imshow('8.) Unknown',unknown)
    cv2.imshow('11.) Contours_Frame',frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break
        
cap.release()
cv2.destroyAllWindows()