import cv2
import mediapipe as mp
import pyautogui
import math

pyautogui.FAILSAFE = False

# Screen size
screen_w, screen_h = pyautogui.size()

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 60)   # higher FPS = less lag

mpHands = mp.solutions.hands
hands = mpHands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mpDraw = mp.solutions.drawing_utils

# ===== Mouse movement tuning (IMPORTANT) =====
prev_x, prev_y = 0, 0
smoothening = 2        # lower = faster response
MOVE_THRESHOLD = 7     # ignore tiny shakes

# Click control
click_down = False

# ===== Scroll control (left hand) =====
prev_left_y = None
SCROLL_THRESHOLD = 20
NORMAL_SCROLL = 300
FAST_SCROLL = 700


def fingers_up(handLms):
    fingers = []

    # Thumb
    fingers.append(handLms.landmark[4].x < handLms.landmark[3].x)

    # Index, Middle, Ring, Pinky
    tips = [8, 12, 16, 20]
    pips = [6, 10, 14, 18]

    for tip, pip in zip(tips, pips):
        fingers.append(handLms.landmark[tip].y < handLms.landmark[pip].y)

    return fingers  # [thumb, index, middle, ring, pinky]


while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape

    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)

    if results.multi_hand_landmarks and results.multi_handedness:
        for handLms, handInfo in zip(results.multi_hand_landmarks,
                                     results.multi_handedness):

            hand_label = handInfo.classification[0].label
            mpDraw.draw_landmarks(img, handLms, mpHands.HAND_CONNECTIONS)

            # ================= RIGHT HAND → MOUSE + CLICK =================
            if hand_label == "Right":
                x_index = int(handLms.landmark[8].x * w)
                y_index = int(handLms.landmark[8].y * h)

                x_thumb = int(handLms.landmark[4].x * w)
                y_thumb = int(handLms.landmark[4].y * h)

                # Convert to screen coordinates
                screen_x = screen_w * x_index / w
                screen_y = screen_h * y_index / h

                dx = screen_x - prev_x
                dy = screen_y - prev_y

                # Dead-zone to remove jitter
                if abs(dx) > MOVE_THRESHOLD or abs(dy) > MOVE_THRESHOLD:
                    curr_x = prev_x + dx / smoothening
                    curr_y = prev_y + dy / smoothening

                    pyautogui.moveTo(curr_x, curr_y)
                    prev_x, prev_y = curr_x, curr_y

                # Click detection (pinch)
                distance = math.hypot(x_index - x_thumb, y_index - y_thumb)

                cv2.line(img, (x_index, y_index),
                         (x_thumb, y_thumb), (0, 255, 0), 2)

                if distance < 35:
                    if not click_down:
                        pyautogui.click()
                        click_down = True
                        cv2.putText(img, "CLICK", (30, 50),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    1, (0, 0, 255), 2)
                else:
                    click_down = False

            # ================= LEFT HAND → SCROLL =================
            if hand_label == "Left":
                fingers = fingers_up(handLms)
                fingers_count = fingers.count(True)

                # FIST → PAUSE SCROLL
                if fingers_count == 0:
                    cv2.putText(img, "SCROLL PAUSED", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (0, 255, 255), 2)
                    prev_left_y = None
                    continue

                wrist_y = int(handLms.landmark[0].y * h)

                if prev_left_y is None:
                    prev_left_y = wrist_y
                    continue

                delta = prev_left_y - wrist_y

                # Two fingers → FAST SCROLL
                speed = FAST_SCROLL if (fingers[1] and fingers[2]) else NORMAL_SCROLL

                if delta > SCROLL_THRESHOLD:
                    pyautogui.scroll(speed)
                    cv2.putText(img, "SCROLL UP", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)

                elif delta < -SCROLL_THRESHOLD:
                    pyautogui.scroll(-speed)
                    cv2.putText(img, "SCROLL DOWN", (30, 90),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1, (255, 0, 0), 2)

                prev_left_y = wrist_y

    cv2.imshow("Hand Gesture Mouse Control", img)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
