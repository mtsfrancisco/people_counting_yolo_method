import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
from tracker import*

model=YOLO('/Users/mtsfrancisco/Documents/cam_detector/yolo_models/yolov8m.pt')


def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

#===========TestVideo.mp4===========#
area1 = [(0, 600), (1920, 600), (1920, 560), (0, 560)]
area2 = [(0, 480), (1920, 480), (1920, 520), (0, 520)]

#=============Camera================ (720, 1280) #
#area1 = [(0,0), (640,0), (640,720), (0, 720)]
#area2 = [(640,0), (1920,0), (1920, 720), (640, 720)]

cap = cv2.VideoCapture('/Users/mtsfrancisco/Documents/cam_detector/media/TestVideo.mp4')
#cap = cv2.VideoCapture(0)
count = 0

tracker = Tracker()

people_entering = {}
people_exiting = {}
entering=set()
exiting=set()

while True:    
    ret,frame = cap.read()
    print(frame.shape)

    if not ret:
        break

    results=model.predict(frame)

    a=results[0].boxes.data
    px=pd.DataFrame(a).astype("float")

    list=[]
    for index,row in px.iterrows():

        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        label=int(row[5])

        if label == 0:
            list.append([x1,y1,x2,y2])

    bbox_ids=tracker.update(list)
    for bbox in bbox_ids:
        x3,y3,x4,y4,id=bbox
        results=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
        if results >= 0:
            people_entering[id] = (x4,y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,0,255),2)
        if id in people_entering:
            results1=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
            if results1 >= 0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)
                cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                entering.add(id)

    #==========people going up==========#
        results2=cv2.pointPolygonTest(np.array(area1,np.int32),((x4,y4)),False)
        if results2 >= 0:
            people_exiting[id] = (x4,y4)
            cv2.rectangle(frame,(x3,y3),(x4,y4),(0,255,0),2)

        if id in people_exiting:
            results3=cv2.pointPolygonTest(np.array(area2,np.int32),((x4,y4)),False)
            if results3 >= 0:
                cv2.rectangle(frame,(x3,y3),(x4,y4),(255,0,255),2)
                cv2.circle(frame,(x4,y4),5,(255,0,255),-1)
                cv2.putText(frame,str(id),(x3,y3),cv2.FONT_HERSHEY_COMPLEX,(0.5),(255,255,255),1)
                exiting.add(id)

            
    cv2.polylines(frame,[np.array(area1,np.int32)],True,(255,0,0),2)
    cv2.polylines(frame,[np.array(area2,np.int32)],True,(255,0,0),2)

    #print(people_entering)
    people_in = (len(entering))
    people_out = (len(exiting))
    cv2.putText(frame,str(people_in),(60,80),cv2.FONT_HERSHEY_COMPLEX,(0.7),(0,0,255),2)
    cv2.putText(frame,str(people_out),(60,140),cv2.FONT_HERSHEY_COMPLEX,(0.7),(255,0,255),2)
    cv2.imshow("RGB", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

