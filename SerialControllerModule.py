import cv2
import os
import HandTrackingModule as htm
import serial
#img должно быть внутри detector

class FingerCounterModule:
    def __init__(self, wCam, hCam, detector):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, wCam) # width
        self.cap.set(4, hCam) # height
        
        self.overlayList = [cv2.imread(f'fingers/{imgPath}') for imgPath in os.listdir("fingers")]
        self.detector = detector
        self.totalFingers = 0
        
        self.ser = serial.Serial('COM4', 9600)
        
        
    
    def run(self):
        while True:
            img = self.cap.read()[1] # Reading the image
            img = cv2.flip(img, 1) # Mirroring the image

            self.detector.findHands(img, False) # Detecting the hand
            self.detector.findPosition(img, False) # Detecting the position of the hand
            self.totalFingers = self.detector.fingersUp(img)

            h, w, c = self.overlayList[self.totalFingers].shape # Extracting the height, width, and channels of the overlay image
            img[0:h, 0:w] = self.overlayList[self.totalFingers] # Overlaying the image on the original image starting from the top left corner

            # self.detector.drawFPS(img)
            for i in self.totalFingers:
                self.ser.write(str(i))
            cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED) # Draw a filled rectangle on the image
            cv2.putText(img, str(self.totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25) # Put text on the image indicating the total number of fingers
             
            # Display the image with the title "Finger Counter" followed by the total number of fingers counted
            cv2.imshow("IMG", img)
            cv2.waitKey(1)
        self.ser.close()
        
app = FingerCounterModule(640, 480, htm.handDetector(detectionCon=0.75, maxHands=1))
app.run()