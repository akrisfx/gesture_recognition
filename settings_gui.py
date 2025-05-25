import tkinter as tk
from tkinter import ttk, messagebox
import json
import threading
from command_executor import CommandExecutor

class SettingsGUI:
    def __init__(self, command_executor=None):
        self.root = tk.Tk()
        self.root.title("Настройки жестов")
        self.root.geometry("700x500")
        self.root.resizable(True, True)
        
        self.executor = command_executor if command_executor else CommandExecutor()
        self.create_widgets()
        self.load_gestures()
    
    def create_widgets(self):
        # Главный фрейм
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Заголовок
        title_label = ttk.Label(main_frame, text="Настройки команд для жестов", font=("Arial", 14, "bold"))
        title_label.pack(pady=(0, 10))
        
        # Фрейм для таблицы
        table_frame = ttk.Frame(main_frame)
        table_frame.pack(fill=tk.BOTH, expand=True)
        
        # Таблица жестов
        columns = ('Жест', 'Название', 'Тип', 'Команда', 'Описание')
        self.tree = ttk.Treeview(table_frame, columns=columns, show='headings', height=15)
        
        # Настройка колонок
        self.tree.heading('Жест', text='ID')
        self.tree.heading('Название', text='Название жеста')
        self.tree.heading('Тип', text='Тип')
        self.tree.heading('Команда', text='Команда')
        self.tree.heading('Описание', text='Описание')
        
        self.tree.column('Жест', width=50)
        self.tree.column('Название', width=120)
        self.tree.column('Тип', width=80)
        self.tree.column('Команда', width=150)
        self.tree.column('Описание', width=200)
        
        # Скроллбар
        scrollbar = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Фрейм для кнопок
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Button(button_frame, text="Добавить", command=self.add_gesture).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(button_frame, text="Редактировать", command=self.edit_gesture).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Удалить", command=self.delete_gesture).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="Сохранить", command=self.save_settings).pack(side=tk.RIGHT)
        
        # Информация о жестах
        info_frame = ttk.LabelFrame(main_frame, text="Доступные жесты")
        info_frame.pack(fill=tk.X, pady=(10, 0))
        
        gesture_info = """0: Fist, 1: Index Up, 2: V, 3: Three Fingers, 4: Four Fingers, 5: Open Palm,
6: Thumb Up, 7: Pinky Up, 8: Rock Sign, 9: Spider-Man, 10: OK Sign, 11: German 3, 12: ROCK, 13: Jambo, 14: Four Fingers 2, 15: Middle"""
        
        ttk.Label(info_frame, text=gesture_info, wraplength=650, justify=tk.LEFT).pack(padx=5, pady=5)
    
    def load_gestures(self):
        """Загружает жесты в таблицу"""
        from main import GestureClassifier
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for gesture_id, data in self.executor.config.items():
            gesture_name = GestureClassifier.get_gesture_name(int(gesture_id))
            self.tree.insert('', 'end', values=(
                gesture_id,
                gesture_name,
                data.get('type', 'key'),
                data.get('command', ''),
                data.get('description', '')
            ))
    
    def add_gesture(self):
        self.gesture_dialog()
    
    def edit_gesture(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите жест для редактирования")
            return
        
        item = self.tree.item(selected[0])
        values = item['values']
        self.gesture_dialog(values[0], values[2], values[3], values[4])
    
    def delete_gesture(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Предупреждение", "Выберите жест для удаления")
            return
        
        item = self.tree.item(selected[0])
        gesture_id = item['values'][0]
        if messagebox.askyesno("Подтверждение", f"Удалить команду для жеста {gesture_id}?"):
            if gesture_id in self.executor.config:
                del self.executor.config[gesture_id]
                self.executor.save_config()
                self.load_gestures()
                messagebox.showinfo("Успех", "Жест удален!")
    
    def gesture_dialog(self, gesture_id='', cmd_type='key', command='', description=''):
        """Диалог для добавления/редактирования жеста"""
        dialog = tk.Toplevel(self.root)
        dialog.title("Настройка команды для жеста")
        dialog.geometry("500x600")
        dialog.resizable(False, False)
        dialog.grab_set()  # Модальное окно
        
        # Основной фрейм
        main_frame = ttk.Frame(dialog)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # ID жеста
        ttk.Label(main_frame, text="ID жеста (0-15):").pack(anchor=tk.W, pady=(0, 5))
        gesture_entry = ttk.Entry(main_frame, width=10)
        gesture_entry.pack(anchor=tk.W, pady=(0, 10))
        gesture_entry.insert(0, gesture_id)
        
        # Тип команды
        ttk.Label(main_frame, text="Тип команды:").pack(anchor=tk.W, pady=(0, 5))
        type_var = tk.StringVar(dialog, value=cmd_type)
        type_frame = ttk.Frame(main_frame)
        type_frame.pack(anchor=tk.W, pady=(0, 10))
        ttk.Radiobutton(type_frame, text="Клавиши", variable=type_var, value="key").pack(side=tk.LEFT)
        ttk.Radiobutton(type_frame, text="Приложение", variable=type_var, value="app").pack(side=tk.LEFT, padx=(20, 0))
        ttk.Radiobutton(type_frame, text="Команда", variable=type_var, value="cmd").pack(side=tk.LEFT, padx=(20, 0))
        
        # Команда
        ttk.Label(main_frame, text="Команда:").pack(anchor=tk.W, pady=(0, 5))
        command_entry = ttk.Entry(main_frame, width=50)
        command_entry.pack(anchor=tk.W, pady=(0, 10))
        command_entry.insert(0, command)
        
        # Описание
        ttk.Label(main_frame, text="Описание:").pack(anchor=tk.W, pady=(0, 5))
        desc_entry = ttk.Entry(main_frame, width=50)
        desc_entry.pack(anchor=tk.W, pady=(0, 15))
        desc_entry.insert(0, description)
        
        # Справка
        help_frame = ttk.LabelFrame(main_frame, text="Примеры команд")
        help_frame.pack(fill=tk.X, pady=(0, 15))
        
        help_text = """Комбинации клавиш:
• ctrl+c, ctrl+v, ctrl+z (копировать, вставить, отменить)
• alt+tab (переключение окон)
• win+d (свернуть все окна)
• space, enter, esc

Приложения:
• notepad, calc, explorer
• chrome, firefox
• cmd, powershell

Консольные команды:
• dir (показать файлы)
• ipconfig (сетевая информация)
• echo Hello World (вывести текст)
• shutdown /s /t 60 (выключение через минуту)"""
        
        ttk.Label(help_frame, text=help_text, justify=tk.LEFT).pack(padx=10, pady=5)
        
        # Кнопки
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill=tk.X, pady=(15, 0))
        
        def save_gesture():
            new_gesture_id = gesture_entry.get().strip()
            new_command = command_entry.get().strip()
            new_description = desc_entry.get().strip()
            new_type = type_var.get()
            
            if not new_gesture_id or not new_command:
                messagebox.showerror("Ошибка", "Заполните ID жеста и команду")
                return
            
            try:
                int(new_gesture_id)  # Проверяем что это число
            except ValueError:
                messagebox.showerror("Ошибка", "ID жеста должен быть числом")
                return
            
            self.executor.config[new_gesture_id] = {
                'type': new_type,
                'command': new_command,
                'description': new_description
            }
            
            # Сохраняем и перезагружаем конфигурацию
            self.executor.save_config()
            if hasattr(self.executor, 'reload_config'):
                self.executor.reload_config()
            
            self.load_gestures()
            dialog.destroy()
            messagebox.showinfo("Успех", "Жест успешно сохранен и применен!")
          # Размещаем кнопки
        ttk.Button(button_frame, text="Сохранить", command=save_gesture).pack(side=tk.LEFT)
        ttk.Button(button_frame, text="Отмена", command=dialog.destroy).pack(side=tk.LEFT, padx=(10, 0))


    def save_settings(self):
        """Сохраняет настройки в файл и уведомляет основную программу"""
        self.executor.save_config()
        
        # Уведомляем основную программу о необходимости перезагрузки конфигурации
        if hasattr(self.executor, 'reload_config'):
            self.executor.reload_config()
        
        messagebox.showinfo("Успех", "Настройки сохранены! Изменения применены к основной программе.")
    
    def run(self):
        self.root.mainloop()

def open_settings_window(command_executor=None):
    """Функция для запуска окна настроек в отдельном потоке"""
    def run_gui():
        app = SettingsGUI(command_executor)
        app.run()
    
    # Запускаем GUI в отдельном потоке, чтобы не блокировать основную программу
    gui_thread = threading.Thread(target=run_gui, daemon=True)
    gui_thread.start()
    return gui_thread

if __name__ == "__main__":
    app = SettingsGUI()
    app.run()
