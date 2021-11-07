import math
import numpy as np
import cv2

#Changeable options:
#Object to tack (Object ID)
obj = 67
#Object tracker from cv2 library
tracker = cv2.TrackerKCF_create()
#Euclidean max distance for object tracking
euc = 100
#Number of frame after which we considere the tracked object to be lost
lostFrame = 10

initBB = None
optFlow = None

detectedObject = None
lostObject=0

# initialize a tracker object from cv2 library
#Options possible:
tracker = cv2.TrackerKCF_create()
#Define our COCO Yolo preloaded model with its weights and class names (80 classes available)
labelsPath = "coco.names"
weightsPath = "yolov3.weights"
configPath = "yolo.cfg"
#Loading the Yolo V3 netwrok
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)


# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )
# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


#Define the video source, in this case webcam and launch it
vcap = cv2.VideoCapture(0)

(ret, initFrame) = vcap.read()

initFrameBW = cv2.cvtColor(initFrame, cv2.COLOR_BGR2GRAY)
extractFeatures = cv2.goodFeaturesToTrack(initFrameBW, mask = None, **feature_params)
# Create a mask image for drawing purposes
mask = np.zeros_like(initFrame)

#Kepp looking as the webcam is capturing images
while vcap.isOpened():
    #initialise MMS data
    boxes = []
    confidences = []
    #get the webcam data as it is filming
    (ret, frame) = vcap.read()
    #get video Hight and width for text display
    W = vcap.get(3)
    H = vcap.get(4)

    #display instructions
    cv2.putText(frame, "Press 'p' to pick your own object", (20, int(H - 60)),cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255), 1)
    cv2.putText(frame, "Once selected, press ENTER",(20, int(H - 35)),cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255), 1)
    cv2.putText(frame, "to exit, press 'e', to go back to yolo 'y'",(20, int(H - 10)),cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255), 1)

    #If own select box has been initialized
    if initBB is not None:
        #get data about tracker success and object bounding box information
        (success, box) = tracker.update(frame)
        #if we are succesfully tracking the object
        if success:
            (x, y, w, h) = [int(v) for v in box]
            #display object bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h),(0, 255, 0), 2)

    #If user didn't select own object, use YOLO to detect object
    else:
        #get the layer info from the yolo network to get the clases after
        layer = net.getLayerNames()
        layer = [layer[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        #create blob with our input image
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        net.setInput(blob)
        #call the forward method of our network
        layerOutputs = net.forward(layer)

        #loop through each node of the output layer
        for node in layerOutputs:
            for data in node:
                #get accuracy and name of each yolo element
                acc = data[5:]
                className = np.argmax(acc)
                confidence = acc[className]
                #if we have a confidence of more than 60% and the selected yolo object
                if confidence > 0.6 and className == obj:
                    #get the object bounding box
                    box = data[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    (x,y,w,h)=(centerX - int(width / 2), centerY - int(height / 2), centerX + int(width / 2),
                                    centerY + int(height / 2))
                    #add the box to the boxes array as single object can get multiple boxes
                    boxes.append([x, y, int(width), int(height)])
                    #for each box we want to keep track of it confidence for NMM later
                    confidences.append(float(confidence))

        #NMS we want to keep only the box with the highest confidence
        if len(boxes) > 0:
            lostObject = 0
            t = confidences[0]
            #print(t)
            # we set the box coordonate to the first detected box by default
            (x, y) = (boxes[0][0], boxes[0][1])
            (w, h) = (boxes[0][2], boxes[0][3])
            #Loop thourhg all the bounding boxes
            for i in range(len(boxes)):
                u = confidences[i]
                if detectedObject is None:
                    #If it's the initial object detected, we just get the highest confidence one
                    if u > t:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
                else:
                    #Else we want to the highest confidence one but also withing the distance threshold
                    newbox = (boxes[i][0], boxes[i][1])
                    #Get the euclidean distance withwee new and old box
                    dist = math.sqrt(sum([(a - b) ** 2 for a, b in zip(detectedObject, newbox)]))
                    if u > t and dist < euc:
                        (x, y) = (boxes[i][0], boxes[i][1])
                        (w, h) = (boxes[i][2], boxes[i][3])
            #If we are not already tracking an object we don't need any distance verification
            if detectedObject is None:
                detectedObject = (x, y)
                cv2.putText(frame, "New Object Detected", (int(W / 2), int(H / 2)),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255), 1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            #Else we want to make sure we are tracking the same Object
            else:
                newObject = (x, y)
                #Calculate the Euclidean distance
                distance = math.sqrt(sum([(a - b) ** 2 for a, b in zip(detectedObject, newObject)]))
                #print("Euclidean distance between two rectangles ", distance)
                #Make sure it's within our tolerated distance
                if distance < euc:
                    detectedObject = newObject
                    #Draw the object bounding box
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        #Check if we lost the object we were tracking
        else:
            if detectedObject is not None:
                lostObject = lostObject+1
                # if the tracked object doesn't show up for a certain amout of frame we just dump it
                if lostObject >= lostFrame:
                    detectedObject = None
                    cv2.putText(frame, "Object Lost", (int(W/2), int(H/2)),
                                cv2.FONT_HERSHEY_COMPLEX_SMALL, .7, (255, 255, 255), 1)

    #show the image as the webcam is filming
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    #if the user presses 'p' to selected own object
    if key == ord("p"):
        #initialise the bounding box with the region of Interested
        initBB = cv2.selectROI("Frame", frame, fromCenter=False,showCrosshair=True)
        #initialize the tracker with the defined bounding box
        tracker.init(frame, initBB)

    #if the user presses 'y' set the initial bounding box to None so we switch back to YOLO
    if key == ord("y"):
        initBB = None
        optFlow = None

    #if the user presses 'y' set the initial bounding box to None so we switch back to YOLO
    if key == ord("f"):
        optFlow = 1

    #if the user presses 'e' break the while loop to exit
    if key == ord("e"):
        break

vcap.release()
cv2.destroyAllWindows()