import cv2
import mediapipe as mp
import math
import time

class HandDetector():
    def __init__(self, mode=False, maxhands=1, modelC = 1, detection_confidence=0.5, tracking_confidence=0.5) -> None:
        self.mode = mode
        self.maxhands = maxhands
        self.modelC = modelC
        self.detection_confidence = float(detection_confidence)
        self.tracking_confidence = float(tracking_confidence)

        self.mphands = mp.solutions.hands
        self.hands = self.mphands.Hands(self.mode, self.maxhands, self.modelC,
                                        self.detection_confidence, self.tracking_confidence)
        self.tip_ids = [4, 8, 12, 16, 20]
        self.mpdraw = mp.solutions.drawing_utils
    
    def FindHands(self, img, draw=True):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(img_rgb)

        if self.results.multi_hand_landmarks:
            for hand_landmarks in self.results.multi_hand_landmarks:
                if draw:
                    self.mpdraw.draw_landmarks(img, hand_landmarks, self.mphands.HAND_CONNECTIONS)

        return img

    def FindPositionOriginal(self, hand_no=0):
        self.lm_list_o = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand_no]        
            for lm in myhand.landmark:
                self.lm_list_o.append([lm.x, lm.y])

        return self.lm_list_o

    def FindPosition(self, img, hand_no=0, draw=True):
        xmin = float('inf')
        xmax = float('-inf')
        ymin = float('inf')
        ymax = float('-inf')

        self.lm_list = []

        if self.results.multi_hand_landmarks:
            myhand = self.results.multi_hand_landmarks[hand_no]
            for id, lm in enumerate(myhand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                self.lm_list.append([id, cx, cy])
                xmin, xmax = min(xmin, cx), max(xmax, cx)
                ymin, ymax = min(ymin, cy), max(ymax, cy)
        
            if draw:
                cv2.rectangle(img, (xmin - 20, ymin-20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
                # cv2.putText(img, f"X:{self.lm_list[8][1]}", (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
                # cv2.putText(img, f"Y:{self.lm_list[8][2]}", (10, 100), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

        return self.lm_list
    
    def FingersUp(self):
        fingers = []
        if self.lm_list[self.tip_ids[0]][1] < self.lm_list[self.tip_ids[0] - 1][1]:
            fingers.append(1)
        else:
            fingers.append(0)

        for id in range(1,5):
            if self.lm_list[self.tip_ids[id]][2] < self.lm_list[self.tip_ids[id] - 3][2]:
                fingers.append(1)
            else:
                fingers.append(0)
        
        return fingers
    
    def FindDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lm_list[p1][1:]
        x2, y2 = self.lm_list[p2][1:]
        cx, cy = (x1 + x2) // 2 , (y1 + y2) // 2
        length = math.hypot(x2 - x1, y2 - y1)

        if draw:
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        return length
    

def main():
    cap = cv2.VideoCapture(0) # Initialize webcam feed
    detector = HandDetector()  # Initialize the HandDetector class

    while True:
        ret, frame = cap.read()  # Read each frame from the webcam

        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.FindHands(frame)  # Find hands in the frame
        lm_list_o = detector.FindPositionOriginal()  # Get landmark positions

        if lm_list_o:  # If hand landmarks detected
            lm_list_o = detector.FindPositionOriginal()

        cv2.imshow("Frame", frame)  # Display the frame

        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit the loop
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

    

                



    

