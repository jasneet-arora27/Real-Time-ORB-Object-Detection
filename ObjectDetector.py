import cv2
import numpy as np
import os

# Path to the directory containing images
path = './Images'

# Create an ORB (Oriented FAST and Rotated BRIEF) object
orb = cv2.ORB_create(nfeatures=1000)

# Import images
images = []
classNames = []
myList = os.listdir(path)
print(myList)
print('Total Classes Detected', len(myList))
for cl in myList:
    # Read images in grayscale
    imgCurr = cv2.imread(f'{path}/{cl}', 0)
    images.append(imgCurr)
    # Remove file extension and add to classNames list
    classNames.append(os.path.splitext(cl)[0])
print(classNames)

# Function to find descriptors of input images
def findDes(images):
    desList = []
    for img in images:
        # Detect and compute keypoints and descriptors using ORB
        kp, des = orb.detectAndCompute(img, None)
        desList.append(des)
    return desList

# Function to find the ID of the input image based on the descriptors
def findID(img, desList, thres=15):
    kp2, des2 = orb.detectAndCompute(img, None)
    bf = cv2.BFMatcher()
    matchList = []
    finalValue = -1
    try:
        for des in desList:
            matches = bf.knnMatch(des, des2, k=2)
            good = []
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good.append([m])
            matchList.append(len(good))
    except:
        pass
    if len(matchList) != 0:
        if max(matchList) > thres:
            finalValue = matchList.index(max(matchList))
    return finalValue

# Find descriptors of all input images
desList = findDes(images)
print(len(desList))

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Main loop for real-time image classification
while True:
    success, img2 = cap.read()
    imgOrig = img2.copy()
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    
    id = findID(img2, desList)
    if id != -1:
        cv2.putText(imgOrig, classNames[id], (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 1)
    
    cv2.imshow('Object', imgOrig)
    
    # Check for the 'q' key to break out of the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all OpenCV windowsṆ
cap.release()
cv2.destroyAllWindows()