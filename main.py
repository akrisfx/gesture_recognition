import cv2
import mediapipe as mp
import time
import threading
import argparse
import os
from command_executor import CommandExecutor

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
        8: ('Rock Sign', [0, 1, 0, 0, 1]),
        9: ('Spider-Man', [1, 1, 0, 0, 1]),
        10: ('OK Sign', [1, 0, 1, 1, 1]),
        11: ('German 3 Fingers', [1, 1, 1, 0, 0]),
        12: ('ROCK', [0, 1, 1, 0, 1]),
        13: ('Jambo', [1, 0, 0, 0, 1]),
        14: ('Four Fingers 2', [1, 1, 1, 1, 0]),
        15: ('Middle', [0, 0, 1, 0, 0])
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
    При этом, любой отход от этой последовательности сбрасывает её.
    Также жест должен удерживаться не менее 200 мс, чтобы считаться валидным.'''
    def __init__(self):
        self.sequence = []
        self.valid_sequence = [5, 0]  # Open Palm -> Fist
        self.ready = False
        self.last_gest = -1
        self.gesture_start_time = time.time()  # Инициализируем текущим временем вместо None
        self.hold_time_ms = 200  # Минимальное время удержания жеста (мс)
    
    def update(self, gesture_idx):
        current_time = time.time()
        
        # Проверка удержания жеста
        if gesture_idx != self.last_gest:
            # Жест изменился, сбрасываем таймер
            self.gesture_start_time = current_time
            self.last_gest = gesture_idx
            held_long_enough = False
        else:
            # Проверяем, достаточно ли долго держится жест
            # Защита от None значения в self.gesture_start_time
            if self.gesture_start_time is None:
                self.gesture_start_time = current_time
                held_long_enough = False
            else:
                held_long_enough = (current_time - self.gesture_start_time) * 1000 >= self.hold_time_ms
        
        if not self.ready:
            # Ожидаем следующий жест из валидной последовательности
            if len(self.sequence) < len(self.valid_sequence):
                expected = self.valid_sequence[len(self.sequence)]
                if gesture_idx == expected and held_long_enough:
                    # Если жест совпал с ожидаемым и держится достаточно долго, добавляем в последовательность
                    self.sequence.append(gesture_idx)
                    # Если вся последовательность показана, устанавливаем флаг готовности
                    if len(self.sequence) == len(self.valid_sequence):
                        self.ready = True
                elif gesture_idx != expected and gesture_idx != self.last_gest:
                    # Если показан неожиданный жест, сбрасываем последовательность
                    self.sequence = []
        else:
            # После успешной последовательности принимаем любой жест, кроме Open Palm и Fist
            if gesture_idx not in self.valid_sequence and gesture_idx != -1 and held_long_enough:
                # Сброс последовательности и флага готовности
                valid_gesture = gesture_idx
                self.sequence = []
                self.ready = False
                return valid_gesture  # Возвращаем валидный жест
            elif gesture_idx == self.valid_sequence[0] and held_long_enough:
                # Если снова показан Open Palm и держится достаточно долго — сбрасываем и начинаем заново
                self.sequence = [gesture_idx]
                self.ready = False
        return None

def parse_arguments():
    """Разбор аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Система распознавания жестов')
    parser.add_argument('--our_mp', action='store_true', 
                        help='Использовать нашу обученную YOLO модель вместо MediaPipe')
    return parser.parse_args()

def create_detector(use_our_model=False):
    """Создает детектор в зависимости от выбранной модели"""
    if use_our_model:
        # Проверяем наличие файла модели
        if not os.path.exists('./best.pt'):
            print("Ошибка: Файл модели './best.pt' не найден!")
            print("Переключаюсь на MediaPipe...")
            use_our_model = False
        else:
            try:
                from ultralytics import YOLO
                print("Инициализация YOLO модели...")
                return YOLO('./best.pt').to('cpu'), 'yolo'
            except ImportError:
                print("Ошибка: ultralytics не установлен!")
                print("Установите: pip install ultralytics")
                print("Переключаюсь на MediaPipe...")
                use_our_model = False
    
    if not use_our_model:
        print("Инициализация MediaPipe...")
        hands = mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.8,
            min_tracking_confidence=0.5
        )
        return hands, 'mediapipe'

def process_yolo_frame(model, image):
    """Обрабатывает кадр с помощью YOLO модели"""
    # Делаем предсказание
    results = model.predict(
        source=image,
        imgsz=320,
        conf=0.4,
        device='cpu',
        verbose=False
    )
    
    # Проверяем наличие ключевых точек
    if results and results[0].keypoints is not None:
        kps_data = results[0].keypoints.xy[0].cpu().numpy()
        
        # Проверяем что есть хотя бы одна точка
        if len(kps_data) > 0:
            # Схема соединений для 21 точки руки
            HAND_CONNECTIONS = [
                (0, 1, 2, 3, 4),         # Большой палец
                (0, 5, 6, 7, 8),         # Указательный палец
                (0, 9, 10, 11, 12),      # Средний палец
                (0, 13, 14, 15, 16),     # Безымянный палец
                (0, 17, 18, 19, 20)      # Мизинец
            ]
            
            # Рисуем соединения
            for finger in HAND_CONNECTIONS:
                for i in range(len(finger)-1):
                    try:
                        start = tuple(map(int, kps_data[finger[i]]))
                        end = tuple(map(int, kps_data[finger[i+1]]))
                        cv2.line(image, start, end, (255, 0, 0), 2)
                    except (IndexError, ValueError):
                        continue
            
            # Рисуем точки
            for point in kps_data:
                try:
                    x, y = map(int, point)
                    cv2.circle(image, (x, y), 4, (0, 255, 0), -1)
                except (ValueError, IndexError):
                    continue
            
            # Преобразуем координаты в формат для распознавания жестов
            # Это упрощенная версия - нужно будет адаптировать под вашу логику жестов
            fingers = analyze_yolo_keypoints(kps_data)
            return fingers, True
    
    return None, False

def analyze_yolo_keypoints(kps_data):
    """Анализирует ключевые точки YOLO для определения состояния пальцев"""
    # Это упрощенная версия - вам нужно будет адаптировать под вашу модель
    # Индексы для кончиков пальцев в YOLO модели (может отличаться)
    tips_ids = [4, 8, 12, 16, 20]  # Большой, указательный, средний, безымянный, мизинец
    
    if len(kps_data) < 21:
        return [0, 0, 0, 0, 0]  # Если недостаточно точек
    
    fingers = []
    
    try:
        # Большой палец (сравниваем x координаты)
        if kps_data[4][0] > kps_data[3][0]:  # Кончик больше предыдущей точки по x
            fingers.append(1)
        else:
            fingers.append(0)
            
        # Остальные пальцы (сравниваем y координаты)
        for i in [8, 12, 16, 20]:  # Указательный, средний, безымянный, мизинец
            if i < len(kps_data) and i-2 < len(kps_data):
                if kps_data[i][1] < kps_data[i-2][1]:  # Кончик выше чем предыдущая точка
                    fingers.append(1)
                else:
                    fingers.append(0)
            else:
                fingers.append(0)
                
    except (IndexError, ValueError):
        return [0, 0, 0, 0, 0]
    
    return fingers

# Инициализация захвата видео
cap = cv2.VideoCapture(0)
prev_time = 0

# Создаем исполнитель команд
command_executor = CommandExecutor()

# Переменные для отслеживания изменений конфигурации
last_config_check = time.time()
config_check_interval = 1.0  # Проверяем файл конфигурации каждую секунду

def open_settings():
    """Открывает окно настроек"""
    try:
        from settings_gui import SettingsGUI
        settings = SettingsGUI()
        settings.run()
    except Exception as e:
        print(f"Ошибка открытия настроек: {e}")

def main():
    """Основная функция программы"""
    global prev_time, last_config_check
    
    # Разбираем аргументы командной строки
    args = parse_arguments()
    
    # Создаем детектор
    detector, detector_type = create_detector(args.our_mp)
    
    print(f"Используется детектор: {detector_type}")
    if detector_type == 'yolo':
        print("Запуск с YOLO моделью (наша обученная модель)")
    else:
        print("Запуск с MediaPipe (стандартная модель)")
    
    validator = GestureValidator()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        gesture_idx = -1
        gesture = 'No hand'
        image = frame.copy()
        if detector_type == 'yolo':
            # Обработка с YOLO
            fingers, hand_detected = process_yolo_frame(detector, image)
            if hand_detected:
                gesture_idx, gesture = GestureClassifier.classify(fingers)
        else:
            # Обработка с MediaPipe
            # Переводим изображение в RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = detector.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    fingers = get_fingers_status(hand_landmarks)
                    gesture_idx, gesture = GestureClassifier.classify(fingers)
        
        # FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        
        # Проверяем, нужно ли перезагрузить конфигурацию
        if curr_time - last_config_check > config_check_interval:
            try:
                config_file = command_executor.config_file
                if os.path.exists(config_file):
                    config_mtime = os.path.getmtime(config_file)
                    if not hasattr(command_executor, '_last_config_mtime'):
                        command_executor._last_config_mtime = config_mtime
                    elif config_mtime > command_executor._last_config_mtime:
                        print("Конфигурация изменена, перезагружаем...")
                        command_executor.reload_config()
                        command_executor._last_config_mtime = config_mtime
            except Exception as e:
                print(f"Ошибка проверки конфигурации: {e}")
            last_config_check = curr_time
        
        validated_gesture = validator.update(gesture_idx)        
        if validated_gesture is not None:
            gesture_idx = validated_gesture
            gesture = GestureClassifier.get_gesture_name(gesture_idx)
            print(gesture, gesture_idx)
            # ВЫПОЛНЯЕМ КОМАНДУ ДЛЯ ЖЕСТА!
            command_executor.execute_gesture_command(gesture_idx)
        else:
            gesture_idx = 0
            gesture = 'pending'
        
        # Отображаем информацию
        model_info = "YOLO" if detector_type == 'yolo' else "MediaPipe"
        cv2.putText(image, f'Model: {model_info}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.putText(image, f'FPS: {fps:.2f}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(image, f'Gesture: {gesture} (#{gesture_idx})', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        cv2.putText(image, 'Press S for Settings, ESC to exit', (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        cv2.imshow('Hand Gesture Recognition', image)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:  # ESC
            break
        elif key == ord('s') or key == ord('S'):  # S for settings
            open_settings()

if __name__ == "__main__":
    main()
    cap.release()
    cv2.destroyAllWindows()
