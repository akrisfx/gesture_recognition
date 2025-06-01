import json
import subprocess
import os
from pynput.keyboard import Key, Controller
import time

class CommandExecutor:
    def __init__(self, config_file='gesture_config.json'):
        self.config_file = config_file
        self.keyboard = Controller()
        self.load_config()
        self.last_executed_time = 0
        self.cooldown = 1.0  # 1 секунда между командами
    
    def load_config(self):
        """Загружает конфигурацию из JSON файла"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            # Создаем конфигурацию по умолчанию
            self.config = {
                "1": {"type": "key", "command": "ctrl+c", "description": "Копировать"},
                "2": {"type": "key", "command": "ctrl+v", "description": "Вставить"},
                "3": {"type": "key", "command": "alt+tab", "description": "Переключить окно"},
                "6": {"type": "key", "command": "space", "description": "Пробел"},
                "7": {"type": "cmd", "command": "dir", "description": "Показать файлы"},
                "8": {"type": "app", "command": "notepad", "description": "Блокнот"},
                "9": {"type": "key", "command": "f11", "description": "Полноэкранный режим"},
                "10": {"type": "app", "command": "calc", "description": "Калькулятор"},
                "11": {"type": "cmd", "command": "ipconfig", "description": "Сетевая информация"},
                "12": {"type": "cmd", "command": "echo Hello from gesture!", "description": "Приветствие"}            }
            self.save_config()
    
    def save_config(self):
        """Сохраняет конфигурацию в JSON файл"""
        with open(self.config_file, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, ensure_ascii=False, indent=2)
    
    def reload_config(self):
        """Перезагружает конфигурацию из файла"""
        try:
            with open(self.config_file, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
            print("Конфигурация перезагружена!")
            return True
        except Exception as e:
            print(f"Ошибка перезагрузки конфигурации: {e}")
            return False
    
    def execute_gesture_command(self, gesture_idx):
        """Выполняет команду для указанного жеста с кулдауном"""
        current_time = time.time()
        if current_time - self.last_executed_time < self.cooldown:
            return  # Слишком рано для следующей команды
        
        gesture_str = str(gesture_idx)
        if gesture_str not in self.config:
            return
        
        command_data = self.config[gesture_str]
        command_type = command_data.get("type", "key")
        command = command_data.get("command", "")
        
        print(f"Выполняю команду: {command_data.get('description', command)}")
        self.last_executed_time = current_time
        
        if command_type == "key":
            self._execute_key_combination(command)
        elif command_type == "app":
            self._execute_application(command)
        elif command_type == "cmd":
            self._execute_command(command)
    
    def _execute_key_combination(self, key_combo):
        """Выполняет нажатие клавиш"""
        keys = key_combo.lower().split('+')
        pressed_keys = []
        
        try:
            # Нажимаем все клавиши
            for key in keys:
                if key == 'ctrl':
                    self.keyboard.press(Key.ctrl)
                    pressed_keys.append(Key.ctrl)
                elif key == 'alt':
                    self.keyboard.press(Key.alt)
                    pressed_keys.append(Key.alt)
                elif key == 'shift':
                    self.keyboard.press(Key.shift)
                    pressed_keys.append(Key.shift)
                elif key == 'win' or key == 'cmd':
                    self.keyboard.press(Key.cmd)
                    pressed_keys.append(Key.cmd)
                elif key == 'tab':
                    self.keyboard.press(Key.tab)
                    pressed_keys.append(Key.tab)
                elif key == 'space':
                    self.keyboard.press(Key.space)
                    pressed_keys.append(Key.space)
                elif key == 'enter':
                    self.keyboard.press(Key.enter)
                    pressed_keys.append(Key.enter)
                elif key == 'esc':
                    self.keyboard.press(Key.esc)
                    pressed_keys.append(Key.esc)
                elif key in ['f1', 'f2', 'f3', 'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12']:
                    func_key = getattr(Key, key)
                    self.keyboard.press(func_key)
                    pressed_keys.append(func_key)
                else:
                    self.keyboard.press(key)
                    pressed_keys.append(key)
            
            time.sleep(0.05)  # Небольшая задержка
            
            # Отпускаем все клавиши в обратном порядке
            for key in reversed(pressed_keys):
                self.keyboard.release(key)
                
        except Exception as e:
            print(f"Ошибка выполнения команды: {e}")
    
    def _execute_application(self, app_name):
        """Запускает приложение"""        
        try:
            subprocess.Popen(app_name, shell=True)
        except Exception as e:
            print(f"Ошибка запуска приложения {app_name}: {e}")
    
    def _execute_command(self, command):
        """Выполняет консольную команду"""
        try:
            print(f"Выполняю команду: {command}")
            
            # Определяем кодировку в зависимости от команды
            encoding = 'cp1251'  # Для Windows команд
            
            # Выполняем команду с правильной кодировкой
            result = subprocess.run(command, shell=True, capture_output=True, text=True, 
                                  encoding=encoding, timeout=10)
            
            if result.stdout:
                print(f"Результат: {result.stdout.strip()}")
            if result.stderr:
                print(f"Ошибка: {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"Команда '{command}' превысила время ожидания (10 сек)")
        except Exception as e:
            print(f"Ошибка выполнения команды '{command}': {e}")
