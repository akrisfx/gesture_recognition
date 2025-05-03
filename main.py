import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

def get_fingers_status(hand_landmarks):
    # Индексы точек для кончиков пальцев и их соседей
    tips_ids = [4, 8, 12, 16, 20]
    fingers = []
    # Большой палец
    if hand_landmarks.landmark[tips_ids[0]].x < hand_landmarks.landmark[tips_ids[0] - 1].x:
        fingers.append(1)
    else:
        fingers.append(0)
    # Остальные пальцы
    for i in range(1, 5):
        if hand_landmarks.landmark[tips_ids[i]].y < hand_landmarks.landmark[tips_ids[i] - 2].y:
            fingers.append(1)
        else:
            fingers.append(0)
    return fingers

def gesture_name(fingers):
    # Примеры простых жестов
    if fingers == [0, 1, 0, 0, 0]:
        return 'Index finger'
    elif fingers == [0, 1, 1, 0, 0]:
        return 'V 2-3'
    elif fingers == [0, 1, 1, 1, 1]:
        return '2-5'
    elif fingers == [1, 1, 1, 1, 1]:
        return 'Open hand'
    elif fingers == [0, 0, 0, 0, 0]:
        return 'Closed hand'
    else:
        return 'Unknown'

# Инициализация захвата видео
cap = cv2.VideoCapture(0)
prev_time = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Переводим изображение в RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        gesture = 'No hand'
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = get_fingers_status(hand_landmarks)
                gesture = gesture_name(fingers)
                print('Landmark 0:', hand_landmarks.landmark[0])

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Жест: {gesture}', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
