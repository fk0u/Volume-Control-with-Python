import cv2
import time
import pyautogui
import math
import numpy as np
import mediapipe as mp
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume, ISimpleAudioVolume

# Pop Up untuk memberitahu program akan segera di mulai
pyautogui.alert('Program about to start')

# Waktu Delay
time.sleep(0.5)

# Inisialisasi MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

# Inisialisasi OpenCV
cap = cv2.VideoCapture(0)
cap.set(3, 640)  # Lebar frame
cap.set(4, 480)  # Tinggi frame

# Mencari jari telunjuk (ID 8) dan jari jempol (ID 4)
finger_ids = [4, 8]

def set_volume(volume):
    devices = AudioUtilities.GetSpeakers()
    interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
    volume_obj = cast(interface, POINTER(IAudioEndpointVolume))
    volume_obj.SetMasterVolumeLevelScalar(volume / 100, None)

# Mencari jari
with mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        frame = cv2.flip(frame, 1)  # Membalikkan frame secara horizontal
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Mendeteksi jari
        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmarks in results.multi_hand_landmarks:
                # Mengambil posisi jari telunjuk dan jari jempol
                finger_points = []
                for id in finger_ids:
                    lm = landmarks.landmark[id]
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    finger_points.append((cx, cy))

                # Menggambar garis antara jari telunjuk dan jari jempol
                cv2.line(frame, finger_points[0], finger_points[1], (255, 0, 255), 3)

                # Mengukur jarak antara dua jari
                length = math.hypot(finger_points[1][0] - finger_points[0][0],
                                    finger_points[1][1] - finger_points[0][1])

                # Skala volume sesuai dengan jarak
                volume = np.interp(length, [20, 200], [0, 100])

                # Mengatur volume suara menggunakan pycaw
                set_volume(volume)

                # Menampilkan volume pada layar
                cv2.putText(frame, f'Volume: {int(volume)}%', (40, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Volume Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
