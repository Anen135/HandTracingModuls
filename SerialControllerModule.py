import cv2
from FingerCounterModule import FingerCounterModule
from HandTrackingModule import handDetector
import serial
#img должно быть внутри detector

class SerialControllerModule:
    def __init__(self, wCam, hCam, detector):
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, wCam) # width
        self.cap.set(4, hCam) # height
        self.counter = FingerCounterModule(detector)
        self.totalFingers = [0,0,0,0,0]
        while True:
            try:
                print("Enter COM Port.\nExample: COM4\n")
                self.ser = serial.Serial(input('COM Port: '), 9600)
                break
            except Exception as e:
                print(e)
                print("Invalid port. Please try again.")
                continue
    
    def run(self):
        print("Press 'q' to quit")
        while True:
            img = self.cap.read()[1] # Reading the image
            img = cv2.flip(img, 1) # Mirroring the image
            self.counter.detector.findHands(img, False) # Detecting the hand
            self.counter.detector.findPosition(img, False) # Detecting the position of the hand
            self.totalFingers = self.counter.fingersUp(img) if self.detector.lmList else [0,0,0,0,0]
            # self.counter.detector.drawFPS(img)
            self.ser.write(bytearray(self.totalFingers))
            self.counter.drawFingerCount(img)
            # Display the image with the title "Finger Counter" followed by the total number of fingers counted
            cv2.imshow("IMG", img)
            if cv2.waitKey(1) & 0xFF in [27, 113, 233, 81]:
                break  
        self.ser.close()
        cv2.destroyAllWindows()
        self.cap.release()
        
app = SerialControllerModule(640, 480, handDetector(detectionCon=0.75, maxHands=1))
app.run()