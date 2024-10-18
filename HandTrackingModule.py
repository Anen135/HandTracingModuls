import cv2
import mediapipe as mp
import time
import math

# TODO: Добавить адекватные логи

class handDetector():
	def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionCon=0.5, trackCon=0.5):
		"""### Initialize the HandTrackingModule object with specified parameters.

		Parameters:
		    mode (bool): Flag to set the mode of the hand tracking module.
		    maxHands (int): Maximum number of hands to detect.
		    modelComplexity (int): Model complexity of the hand tracking model.
		    detectionCon (float): Minimum detection confidence threshold.
		    trackCon (float): Minimum tracking confidence threshold.

		Returns:
		    None
		"""
		self.mode = mode
		self.maxHands = maxHands
		self.modelComplexity = modelComplexity
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.mpHands = mp.solutions.hands
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
		self.mpDraw = mp.solutions.drawing_utils
		self.tipIds = [4, 8, 12, 16, 20] 
		self.results = None
		self.lmList = []
  
		self.pTime = 0
  
		self.fps = {
			"org": (10, 70),
			"font": 1, # FONT_HERSHEY_PLAIN
			"scale": 3,
			"color": (255, 0, 255),
			"bold": 3
		}
		self.rectangle = {
			"color": (0, 255, 0),
			"bold": 2
		}
		self.circle = {
			"radius": 5,
			"color": (255, 0, 255),
			"bold": -1 # FILLED
		}

	def findHands(self, img, draw=True):
		imgRGB = cv2.cvtColor(img, 4) # BGR -> RGB
		self.results = self.hands.process(imgRGB)
		# print(self.results.multi_hand_landmarks)

		if draw and self.results.multi_hand_landmarks:
			for handLms in self.results.multi_hand_landmarks: self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

	def findPosition(self, img, handNo=0, draw=True):
		xList = []
		yList = []
		xmin = xmax = ymin = ymax = 0
		if self.results.multi_hand_landmarks:
			myHand = self.results.multi_hand_landmarks[handNo]
			for id, lm in enumerate(myHand.landmark):
				#print(id, lm)
				h, w, c = img.shape
				cx, cy = int(lm.x*w), int(lm.y*h)
				xList.append(cx)
				yList.append(cy)
				#print(id, cx, cy)
				self.lmList.append([id, cx, cy])
				if draw and id in self.tipIds:
					cv2.circle(img, (cx, cy), self.circle['radius'], self.circle['color'], self.circle['bold'])
			xmin, xmax, ymin, ymax = min(xList), max(xList), min(yList), max(yList)
			if draw:
				cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), self.rectangle['color'], self.rectangle['bold'])
		return xmin, ymin, xmax, ymax

	def findDistance(self, p1, p2, img, draw=True):
		x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
		x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
		cx, cy = (x1+x2)//2, (y1+y2)//2

		if draw:
			cv2.circle(img, (x1,y1), 15, (255,0,255), cv2.FILLED)
			cv2.circle(img, (x2,y2), 15, (255,0,255), cv2.FILLED)
			cv2.line(img, (x1,y1), (x2,y2), (255,0,255), 3)
			cv2.circle(img, (cx,cy), 15, (255,0,255), cv2.FILLED)

		length = math.hypot(x2-x1, y2-y1)
		return length, img, [x1, y1, x2, y2, cx, cy]

	def fingersUp(self):
		fingers = []
		
		# Thumb
		if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]:
			fingers.append(1)
		else:
			fingers.append(0)

		# 4 Fingers
		for id in range(1,5):
			if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]:
				fingers.append(1) # TODO: Просто преобразуй булевое значение в int
			else:
				fingers.append(0)
		return fingers

	def drawFPS(self, img):
		cTime = time.time()
		cv2.putText(img, str(int(1 / (cTime - self.pTime))), self.fps["org"], self.fps['font'], self.fps['scale'], self.fps['color'], self.fps['bold'])
		self.pTime = cTime
	
	def drawRectangle(self, img, xmin, ymin, xmax, ymax):
		cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), self.rectangle['color'], self.rectangle['bold'])
     

def demo():
	cap = cv2.VideoCapture(0)
	detector = handDetector()
	while True:
		img = cap.read()[1]
		detector.findHands(img)
		detector.findPosition(img)
		# print(bbox)

		detector.drawFPS(img)

		cv2.imshow("Image", img)
		cv2.waitKey(1)


if __name__ == "__main__":
	demo()