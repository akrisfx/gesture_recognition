import cv2
import mediapipe as mp
import time

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

class GestureClassifier:
    GESTURES = {
        0: ('Fist', [0, 0, 0, 0, 0]),
        1: ('Index Up', [0, 1, 0, 0, 0]),
        2: ('V (Two Fingers)', [0, 1, 1, 0, 0]),
        3: ('Three Fingers', [0, 1, 1, 1, 0]),
        4: ('Four Fingers', [0, 1, 1, 1, 1]),
        5: ('Open Palm', [1, 1, 1, 1, 1]),
        6: ('Thumb Up', [1, 0, 0, 0, 0]),
        7: ('Pinky Up', [0, 0, 0, 0, 1]),
        8: ('Rock Sign', [1, 0, 0, 0, 1]),
        9: ('Spider-Man', [1, 1, 0, 0, 1]),
        10: ('OK Sign', [1, 0, 1, 1, 1]),
        11: ('German 3 Fingers', [1, 1, 1, 0, 0]),
        12: ('ROCK', [0, 1, 1, 0, 1]),
        13: ('Jambo', [1, 0, 0, 0, 1]),
        14: ('Four Fingers 2', [1, 1, 1, 1, 0]),
    }

    @classmethod
    def get_gesture_index(cls, fingers):
        for idx, (_, pattern) in cls.GESTURES.items():
            if fingers == pattern:
                return idx
        return -1

    @classmethod
    def get_gesture_name(cls, idx):
        return cls.GESTURES.get(idx, ('Unknown', []))[0]

    @classmethod
    def classify(cls, fingers):
        idx = cls.get_gesture_index(fingers)
        return idx, cls.get_gesture_name(idx)

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

class GestureValidator:
    '''Идея состоит в том, что мы регистрируем срабатывание жеста только при опрделенной последовательности
    жестов. Тоесть чтобы жест был распознан, сначала пользователь должен показать Open Palm, после сжать кулак,
    и после этого показать жест который он хочет показать, тем самым мы избегаем случайного распознавания жестов.
    При этом, любой отход от этой последовательности сбрасывает её.'''
    def __init__(self):
        self.sequence = []
        self.valid_sequence = [5, 0]  # Open Palm -> Fist
        self.ready = False
        self.last_gest = -1
    
    def update(self, gesture_idx):
        if not self.ready:
            # Ожидаем следующий жест из валидной последовательности
            expected = self.valid_sequence[len(self.sequence)]
            if gesture_idx == expected:
                # Если жест совпал с ожидаемым, добавляем в последовательность
                self.sequence.append(gesture_idx)
                self.last_gest = gesture_idx
                # Если вся последовательность показана, устанавливаем флаг готовности
                if len(self.sequence) == len(self.valid_sequence):
                    self.ready = True

            # обработка старого жеста но всё еще валидного
            elif gesture_idx == self.last_gest: 
                # Если жест уже в последовательности, ничего не делаем
                time.sleep(0.1)
                pass
            else:
                # Если жест не совпал, сбрасываем последовательность
                self.sequence = []
        else:
            # После успешной последовательности принимаем любой жест, кроме Open Palm и Fist
            if gesture_idx not in self.valid_sequence and gesture_idx != -1:
                # Сброс последовательности и флага готовности
                self.sequence = []
                self.ready = False
                return gesture_idx  # Возвращаем валидный жест
            elif gesture_idx == self.valid_sequence[0]:
                # Если снова показан Open Palm — сбрасываем и начинаем заново
                self.sequence = [gesture_idx]
                self.ready = False
        return None

# Инициализация захвата видео
cap = cv2.VideoCapture(0)
prev_time = 0

with mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
) as hands:
    validator = GestureValidator()
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

        gesture_idx = -1
        gesture = 'No hand'
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                fingers = get_fingers_status(hand_landmarks)
                gesture_idx, gesture = GestureClassifier.classify(fingers)
                print('Landmark 0:', hand_landmarks.landmark[0])

        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time

        validated_gesture = validator.update(gesture_idx)
        if validated_gesture is not None:
            gesture_idx = validated_gesture
            gesture = GestureClassifier.get_gesture_name(gesture_idx)
        else:
            gesture_idx = 0
            gesture = 'pending'
        cv2.putText(image, f'FPS: {fps:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Gesture: {gesture} (#{gesture_idx})', (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

        cv2.imshow('Hand Gesture Recognition', image)
        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()
