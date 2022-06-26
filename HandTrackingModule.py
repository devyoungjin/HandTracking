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

class handDetector():
    
    def __init__(self,mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        # static_image_mode = False,
        # max_num_hands = 2,
        # min_detection_confidence = 0.5,
        # min_tracking_confidence = 0.5

        self.mpHands = mp.solutions.hands
        #error
        # self.hands = self.mpHands.Hands(self.mode, self.maxHands,
        #                                 self.detectionCon, self.trackCon) #only RGB

        self.hands = self.mpHands.Hands() #only RGB
        self.mpDraw = mp.solutions.drawing_utils
        
        wCam, hCam = 720, 480
        cap.set(3,wCam)
        cap.set(4,hCam)

    def findHands(self,img,draw=True):
        imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)
        #print(results.multi_hand_landmarks)

        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                # for id, lm in enumerate(handLms.landmark):
                #     # print(id,lm)
                #     h,w,c = img.shape
                #     cx,cy = int(lm.x*w),int(lm.y*h) #consider a pixel value 
                #     print(id,cx,cy)

                #     if id == 0: 
                #         cv2.circle(img,(cx,cy),15,(255,0,255),cv2.FILLED)
                if draw:
                    self.mpDraw.draw_landmarks(img,handLms,self.mpHands.HAND_CONNECTIONS) #single hand
        return img


def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img,f'FPS:{int(fps)}',(40,50),cv2.FONT_HERSHEY_COMPLEX, 1,(255,0,0),3)
    
        cv2.imshow("Img",img)
        cv2.waitKey(1) #1minute 

if __name__ =="__main__":
    main()