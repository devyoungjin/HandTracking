import sys
import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)

####### parameter
wCam, hCam = 720, 480
cap.set(3,wCam)
cap.set(4,hCam)
#######


mpHands = mp.solutions.hands
hands = mpHands.Hands() #only RGB
mpDraw = mp.solutions.drawing_utils

pTime = 0



while True:
    success, img = cap.read()

    if img is None:
        print('load failed')
        sys.exit()
        
    imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    #print(results.multi_hand_landmarks)

    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c = img.shape
                cx,cy = int(lm.x*w),int(lm.y*h) #consider a pixel value 
                print(id,cx,cy)

                if id == 0: 
                    cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)

            mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS) #single hand
    

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime
    cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0),3)
  
    cv2.imshow("Img",img)
    cv2.waitKey(1) #1minute 
