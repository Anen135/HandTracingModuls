import cv2
import os
import sys
import HandTrackingModule as htm
from config import distance
#img должно быть внутри detector

class FingerCounterModule:
    def __init__(self, detector):
        if hasattr(sys, "_MEIPASS"):  # PyInstaller создает временную директорию при сборке
            folder_path = os.path.join(sys._MEIPASS, 'fingers')
        else:
            folder_path = 'fingers'  # Для обычного запуска из Python
        self.overlayList = [cv2.imread(os.path.join(folder_path, imgPath)) for imgPath in os.listdir(folder_path)]
        self.detector = detector
        self.totalFingers = 0
    
    def drawFingerCount(self, img):
        cv2.rectangle(img, (20, 225), (170, 425), (0, 255, 0), cv2.FILLED) # Draw a filled rectangle on the image
        cv2.putText(img, str(self.totalFingers), (45, 375), cv2.FONT_HERSHEY_PLAIN, 10, (255, 0, 0), 25) # Put text on the image indicating the total number of fingers
    
    def drawImage(self, img):
        h, w, c = self.overlayList[self.totalFingers].shape # Extracting the height, width, and channels of the overlay image
        img[0:h, 0:w] = self.overlayList[self.totalFingers] # Overlaying the image on the original image starting from the top left corner
    
    def fingersUpCount(self, img):
        Reference = self.detector.findDistance(5, 0, img, colorL=distance['reference']['ColorL'], colorC=distance['reference']['ColorC'])[0]
        rfl = Reference/0.75
        return (self.detector.findDistance(4, 17, img)[0] > Reference / 1.3) + (self.detector.findDistance(8, 0, img)[0] > rfl) + (self.detector.findDistance(12, 0, img)[0] > rfl) + (self.detector.findDistance(16, 0, img)[0] > rfl) + (self.detector.findDistance(20, 0, img)[0] > rfl)
    
    def fingersUp(self, img) -> list:
        Reference = self.findDistance(5, 0, img, colorL=distance['reference']['ColorL'], colorC=distance['reference']['ColorC'])[0]
        rfl = Reference/0.75
        return [self.findDistance(4, 17, img)[0] > Reference / 1.3, self.findDistance(8, 0, img)[0] > rfl, self.findDistance(12, 0, img)[0] > rfl, self.findDistance(16, 0, img)[0] > rfl, self.findDistance(20, 0, img)[0] > rfl]
    
def demo(app):
    cap = cv2.VideoCapture(0)
    cap.set(3, 640) # width
    cap.set(4, 480) # height
    print('Press "q" to quit')
    while True:
        img = cap.read()[1] # Reading the image
        img = cv2.flip(img, 1) # Mirroring the image
        app.detector.findHands(img) # Detecting the hand
        app.detector.findPosition(img) # Detecting the position of the hand

        app.totalFingers = app.fingersUpCount(img) if app.detector.lmList else 0 # Get the total number of fingers counted

        #app.drawImage(img)
        app.drawFingerCount(img)
        app.detector.drawFPS(img)             
        # Display the image with the title "Finger Counter" followed by the total number of fingers counted
        cv2.imshow("IMG", img) 
        # If the window is closed, break the loop
        if cv2.waitKey(1) & 0xFF in [27, 113, 233, 81]:
            break   
    cap.release()  
    cv2.destroyAllWindows()  
        
if __name__ == "__main__":
    demo(FingerCounterModule(htm.handDetector(detectionCon=0.75, maxHands=1)))