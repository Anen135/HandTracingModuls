import cv2
import mediapipe as mp
import time
import math
from config import fps, rectangle, distance

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
		self.mpHands = mp.solutions.hands
		self.mpDraw = mp.solutions.drawing_utils
		self.mpDrawingStyles = mp.solutions.drawing_styles

		self.mode = mode
		self.maxHands = maxHands
		self.modelComplexity = modelComplexity
		self.detectionCon = detectionCon
		self.trackCon = trackCon

		self.results = None
		self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionCon, self.trackCon)
		self.tipIds = [4, 8, 12, 16, 20] 
		self.results = None
		self.lmList = []
		self.pTime = 0

	def findHands(self, img):
		self.results = self.hands.process(cv2.cvtColor(img, 4)) # BGR -> RGB and get the results
		# print(self.results.multi_hand_landmarks)
	def findPosition(self, img, handNo=0):
		self.lmList = []
		if self.results.multi_hand_landmarks:
			xList = []
			yList = []
			myHand = self.results.multi_hand_landmarks[handNo]
			#print(id, lm)
			h, w, c = img.shape
			for id, lm in enumerate(myHand.landmark):
				cx, cy = int(lm.x*w), int(lm.y*h)
				xList.append(cx)
				yList.append(cy)
				#print(id, cx, cy)
				self.lmList.append([id, cx, cy])
			xmin, xmax, ymin, ymax = min(xList), max(xList), min(yList), max(yList)
			return xmin, ymin, xmax, ymax
		else:
			return None

	def findDistance(self, p1, p2, img, draw=distance['draw'], colorC=distance['circle']['color'], colorL=distance['line']['color']):
		x1, y1 = self.lmList[p1][1], self.lmList[p1][2]
		x2, y2 = self.lmList[p2][1], self.lmList[p2][2]
		cx, cy = (x1+x2)//2, (y1+y2)//2
		length = math.hypot(x2-x1, y2-y1)
		if draw: self.drawDistance(img, length, [x1, y1, x2, y2, cx, cy], colorC, colorL)
		return length, [x1, y1, x2, y2, cx, cy]


	def drawDistance(self, img, length, positions, colorC, colorL):
		referenceLine = int(length//25)
		cv2.line(img, (positions[0],positions[1]), (positions[2],positions[3]), colorL, distance['line']['bold'])
		cv2.circle(img, (positions[0],positions[1]), referenceLine, colorC, distance['circle']['bold'])
		cv2.circle(img, (positions[2],positions[3]), referenceLine // 16, colorC, distance['circle']['bold'])
		cv2.circle(img, (positions[4],positions[5]), referenceLine, colorC, distance['circle']['bold'])
	def drawFPS(self, img):
		cTime = time.time()
		cv2.putText(img, str(int(1 / (cTime - self.pTime))), fps["org"], fps['font'], fps['scale'], fps['color'], fps['bold'])
		self.pTime = cTime
	def drawRectangle(self, img, position):
		xmin, ymin, xmax, ymax = position
		cv2.rectangle(img, (xmin-20, ymin-20), (xmax+20, ymax+20), rectangle['color'], rectangle['bold'])
	def drawHandMarks(self, img):
		if self.results.multi_hand_landmarks:
			for landmark in self.results.multi_hand_landmarks:
				self.mpDraw.draw_landmarks(img, landmark, self.mpHands.HAND_CONNECTIONS, self.mpDrawingStyles.get_default_hand_landmarks_style(), self.mpDrawingStyles.get_default_hand_connections_style())
    

def demo(detector):
	cap = cv2.VideoCapture(0)
	while True:
		img = cap.read()[1]
		detector.findHands(img)
		if pos := detector.findPosition(img): detector.drawRectangle(img, pos)
		detector.drawHandMarks(img)
		# print(bbox)

		detector.drawFPS(img)

		cv2.imshow("Image", img)
		if cv2.waitKey(1) & 0xFF in [27, 113, 233, 81]:
			break  
	cap.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	demo(handDetector())