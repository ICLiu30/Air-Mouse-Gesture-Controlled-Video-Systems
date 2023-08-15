import cv2
import mediapipe as mp
import numpy as np
import pyautogui
from module import HandDetector
import collections
import time
import speech_recognition as sr
import threading
import pickle
from tkinter import *
from pathlib import Path

dir = Path(__file__).resolve().parent 

path_to_model = dir / 'datasets' / 'rfclassifier.pkl'

with open(path_to_model, 'rb') as file:
    model = pickle.load(file)

root = Tk()
root.title("Air Control Status")
root.geometry("250x50")

stop_typing_event = threading.Event()
wScr, hScr = pyautogui.size()
cap = cv2.VideoCapture(0)
detector = HandDetector()
wCam, hCam = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
wframeR = 300
hframeR = 150
cursorX_deque = collections.deque(maxlen=5)
cursorY_deque = collections.deque(maxlen=5)
click_delay = 0.4
doubleclick = collections.deque(maxlen=2)
last_click_time = 0
TipPlocX, TipPlocY = wCam // 2, hCam // 2
r = sr.Recognizer()
mode_type = np.zeros((150, 400, 3), dtype=np.uint8)
    
def get_voice_command(retries=1, phrase_time_limit=None):
    command = ''
    with sr.Microphone() as source:

        r.adjust_for_ambient_noise(source, duration=0.5)
        print("Please say something...")
        audio_data = r.listen(source, phrase_time_limit=phrase_time_limit)

        print('Recognizing...')
        for _ in range(retries):
            try:
                command = r.recognize_google(audio_data)
                command = command.lower()
                print("You said: " + command)
                break
            except sr.UnknownValueError:
                print("Google Speech Recognition could not understand audio")
            except sr.RequestError as e:
                print(f"Could not request results from Google Speech Recognition service; {e}")
                time.sleep(1)
    return command

def video_control():
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        frame = detector.FindHands(frame)
        
        lm_list_o = detector.FindPositionOriginal()
        detector.FindPosition(frame)
        
        command_status = 'Unknown'  # Default value
        
        if lm_list_o:
            fingers = detector.FingersUp()
            
            keypoints = []
            for i in range(len(lm_list_o)):
                keypoints.append(lm_list_o[i][0])
                keypoints.append(lm_list_o[i][1])
            
            x = np.array(keypoints).reshape(1, -1)
            
            command_class = model.predict(x)[0]
            command_prob = model.predict_proba(x)[0][model.predict_proba(x)[0].argmax()]

            if command_prob >= 0.6:
                command_status = command_class
            else:
                command_status = 'Unknown'
        
            cv2.rectangle(frame, (0,0), (500, 120), (245, 117, 16), -1)
        
            cv2.putText(frame, f"command: {command_status}", (10, 35), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            cv2.putText(frame, f"probs: {command_prob}", (10, 85), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)

            if command_status == 'pause':
                pyautogui.press('space')
                time.sleep(1)
            elif command_status == 'up':
                pyautogui.press('up')
                time.sleep(0.5)
            elif command_status == 'down':
                pyautogui.press('down')
                time.sleep(0.5)
            elif command_status == 'rewind':
                pyautogui.press('left')
                time.sleep(0.5)
            elif command_status == 'forward':
                pyautogui.press('right')
                time.sleep(0.5)
            
            elif fingers[1:] == [1,1,1,0]:
                voice_command = ''
                cv2.destroyAllWindows()
                return voice_command
                
            
        cv2.imshow('frame', frame)
        cv2.waitKey(1)


def mouse_control(stop_typing_event):
    global last_click_time, doubleclick

    while True:
        ret, frame = cap.read()
        
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        frame = detector.FindHands(frame) 
        lm_list = detector.FindPosition(frame)

        if lm_list:
            TipX, TipY = lm_list[8][1:]
            fingers = detector.FingersUp() 
            cv2.rectangle(frame, (wframeR, hframeR), (wCam - wframeR, hCam - hframeR), (255, 0, 255), 2)
            cv2.putText(frame, str(fingers[0]), (50, 50), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
            
            # Moving and Click
            if fingers[1] == 1 and fingers[2] == 0:
                cursorX = np.interp(TipX, (wframeR, wCam - wframeR), (0, wScr))
                cursorY = np.interp(TipY, (hframeR, hCam - hframeR), (0, hScr))

                # Add the new positions to the deques
                cursorX_deque.append(cursorX)
                cursorY_deque.append(cursorY)

                # Calculate the average positions
                cursorX_avg = sum(cursorX_deque) / len(cursorX_deque)
                cursorY_avg = sum(cursorY_deque) / len(cursorY_deque)

                pyautogui.moveTo(cursorX_avg, cursorY_avg)

                current_time = time.time()

                if lm_list[4][1] > lm_list[2][1] and current_time - last_click_time >= click_delay:
                    doubleclick.append(1)
                    pyautogui.click()
                    last_click_time = current_time
                    if list(doubleclick) == [1, 1]:
                        pyautogui.doubleClick()
                        doubleclick = collections.deque(maxlen=2)
                else:
                    doubleclick.append(0)

            # Scroll
            elif fingers[1:] == [1,1,0,0]:
                doubleclick.append(0)
                if lm_list[8][2] > lm_list[7][2] and lm_list[12][2] > lm_list[11][2]:
                    pyautogui.hotkey('alt', 'down')
                    time.sleep(0.8)
                else:
                    pyautogui.hotkey('alt', 'up')
                    time.sleep(0.8)

            # Stop
            elif fingers[1:] == [1,1,1,0]:
                voice_command = ''
                cv2.destroyAllWindows()
                stop_typing_event.set()
                return voice_command
        
        # Display the frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(1)

def typing(stop_typing_event):
    while not stop_typing_event.is_set():
        print('typing mode activate')
        text = get_voice_command()
        if 'enter' in text.lower():
            text = text.replace('enter', '')
            pyautogui.typewrite(text)

my_label = Label(root, text="Stand By", font=("Arial", 20))
my_label.pack()

get_voice_command(phrase_time_limit=1)

while True:
    stop_typing_event = threading.Event()
    my_label.config(text='Stand By')
    root.update()
    voice_command = get_voice_command()

    if 'mouse control' in voice_command:
        my_label.config(text='Mouse Control Mode')
        root.update()
        thread1 = threading.Thread(target=typing, args=(stop_typing_event,))  # Corrected the thread creation
        thread1.start()
        mouse_control(stop_typing_event)

    elif 'video control' in voice_command:
        my_label.config(text='Video Control Mode')
        root.update()
        video_control()

        pass

    elif voice_command == 'stop':
        break