"""Основной файл проекта

Суть программы: пользователь может рисовать цифры в разрешением 28 * 28, а нейросеть должна предсказать, какое число нарисовано.

RoadMap:
    Интерфейс, где пользователь может рисовать и  последствии это переведётся в картинку 28 * 28 и переведётся в numpy array.
    
    Неросеть: установка весов, чтобы каждый раз не обучать заново
    https://www.machinelearningmastery.ru/save-load-machine-learning-models-python-scikit-learn/


Интерфейс приложения, осуществлённый при помощи библиотеки kivy.

https://kivy.org/doc/stable/api-kivy.graphics.html#module-kivy.graphics

Отсюда взята рисовалка - https://pythonprogramming.net/kivy-drawing-application-tutorial/
"""
import kivy
kivy.require('2.0.0')

import os
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
import pickle

from kivy.core.window import Window
from kivy.graphics import Line
from kivy.utils import get_color_from_hex

from kivy.uix.widget import Widget
from kivy.uix.screenmanager import Screen, ScreenManager

from kivymd.app import MDApp
from kivymd.color_definitions import colors
from kivymd.uix.floatlayout import MDFloatLayout
from kivymd.uix.progressbar import MDProgressBar
from kivymd.uix.button import MDRaisedButton, MDFlatButton
from kivymd.uix.label import MDLabel


Window.size = (WIDTH, HEIGHT) = (800, 650)  # Размеры окна приложения


class Canvas(Widget):
    """Холст, область 504 на 504, расположенная в позиции (0, 146), в которой можно рисовать"""

    def __init__(self, predict_f, **kwargs):
        super().__init__(**kwargs)

        # Меняем размеры и положения холста относительно родителя
        self.size_hint = (504 / WIDTH, 504 / HEIGHT)
        self.pos_hint = {'x': 0, 'y': 146 / HEIGHT}

        # Назначаем функцию предсказания
        self.predict = predict_f

    def on_touch_down(self, touch):
        """Метод, срабатывающий при косании по холсту"""
        # Проверяем входит ли точка касания в область холста
        # touch.pos = [touch.x, touch.y]
        if not self.collide_point(*touch.pos):
            return

        # Взаимодейтвуем схолстом. Начинаем новую линию
        with self.canvas:
            touch.ud["line"] = Line(
                points=(touch.x, touch.y),
                width=10,
            )

    def on_touch_move(self, touch):
        """Метод, срабатывающий при проведении по холсту. Создаёт непрерывную линию во время проведения."""
        # Проверяем входит ли точка прикосновения в данный момент в область холста
        if self.collide_point(*touch.pos):
            touch.ud["line"].points += (touch.x, touch.y)

    def on_touch_up(self, touch):
        """Метод, срабатывающий после отжимания пальца или мыши. В данном случае можно сразу предсказывать число после рисовки, чтобы не нажимать на кнопку.
        """
        self.predict(self)  # Передаём self как instance


'''
class Probabilities(MDGridLayout, MDLabel):
    """Элемент, в котором отображаются предсказания вероятности того или иного числа нейросетью"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Задаём размеры и позици, относительные внутри элемента родителя
        self.size_hint = (296 / WIDTH, 503 / HEIGHT)
        self.pos_hint = {'x': 504 / WIDTH, 'y': 147 / HEIGHT}
        
        #! self.md_bg_color=[1, 1, 1, 1]
        
        # Добавляем ряды в grid layout
        for num in range(10):
            self.add_widget(MDLabel(
                text=f"{num}", 
                theme_text_color="Custom",
                text_color=(255 / 255, 152 / 255, 0, 1),
                
                bold=True,
                font_style="H4",
                size_hint_x=(40 / 296)
            ))
            self.add_widget(MDProgressBar())
            # self.add_widget(Probability(num))
'''


class Probabilities(MDFloatLayout, MDLabel):
    """Элемент, в котором отображаются предсказания вероятности того или иного числа нейросетью"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Задаём размеры и позици, относительные внутри элемента родителя
        self.size_hint = (296 / WIDTH, 503 / HEIGHT)
        self.pos_hint = {'x': 504 / WIDTH, 'y': 147 / HEIGHT}
        # Белый фон
        self.md_bg_color = [1, 1, 1, 1]

        # Добавляем ряды во floatlayout
        for num in range(9, -1, -1):
            # Добавление слоя с цифрой
            self.add_widget(MDLabel(
                text=f"{9 - num}",
                theme_text_color="Custom",
                text_color=(255 / 255, 152 / 255, 0, 1),  # Оранжевый
                # text_color=(0, 0, 0, 1),

                bold=True,
                font_style="H3",
                size_hint=(40 / 296, 48 / 504),
                pos_hint={'x': 60 / WIDTH, 'y': (60 * num + 25) / HEIGHT}
            ))
            # Добавление прогрессбара
            self.add_widget(MDProgressBar(
                max=1,
                size_hint=(180 / 296, 48 / 504),
                pos_hint={'x': 200 / WIDTH, 'y': (60 * num + 35) / HEIGHT}  # 60 - расстояние между строками; 35 - сдвиг от нуля
            ))
            # Добавление процентов
            self.add_widget(MDLabel(
                text="0%",
                theme_text_color="Custom",
                text_color=(0, 0, 0, 0.7),
                halign="right",
                size_hint=(180 / 296, 48 / 504),
                pos_hint={'x': 200 / WIDTH, 'y': (60 * num + 18) / HEIGHT}
            ))


class MainScreen(Screen):
    """Главный экран программы"""

    def __init__(self, **kw):
        super().__init__(**kw)

        # Добавляем холст, с переданной функцией предсказания
        self.ui = Canvas(predict_f=self.predict_canvas)  # self.predict_canvas)
        self.add_widget(self.ui)

        # Добавляем кнопку очищения
        clear_btn = MDRaisedButton(
            text="[color=#ffffff][b]clear[/b][/color]",
            # theme_text_color="Custom",
            font_size='40sp',
            size_hint=(146 / WIDTH, 146 / HEIGHT),
            on_release=self.clear_canvas,
        )
        self.add_widget(clear_btn)

        # Добавляем сохранения холста
        save_btn = MDFlatButton(
            text="[color=#ffffff][b]save[/b][/color]",
            # text_color=(1, 1, 1, 0),
            # theme_text_color="Custom",
            md_bg_color=(1, 1, 1, 1),
            font_size='40sp',
            
            size_hint=(358 / WIDTH, 146 / HEIGHT),
            pos_hint={'x': 147 / WIDTH},
            
            on_release=self.save_image,
        )
        self.add_widget(save_btn)

        # Добавляем панель вероятностей
        self.probabilities = Probabilities()
        self.add_widget(self.probabilities)

        # Добавляем слой с отображением конечного предказания нейросети
        self.prediction_lbl = MDLabel(
            theme_text_color="Custom",
            text_color=(255 / 255, 152 / 255, 0, 1),
            md_bg_color=[1, 1, 1, 1],

            text="?",
            font_style="H1",
            bold=True,

            halign="center",
            size_hint=(296 / WIDTH, 146 / HEIGHT),
            pos_hint={'x': 504 / WIDTH}
        )
        self.add_widget(self.prediction_lbl)

        # Загружаем обученную нейросеть
        self.predictor: "MLPClassifier" = pickle.load(open("models/model a(0.976) hl(100_100_50) i(175).sav", "rb"))

    def clear_canvas(self, instance):
        """Очистка холста, вероятностей и конечного предсказания"""
        self.ui.canvas.clear()

        for pb in self.probabilities.children[-2::-3]:
            pb.value = 0

        for p in self.probabilities.children[-3::-3]:
            p.text = "0%"

        self.prediction_lbl.text = "?"

    def predict_canvas(self, instance):
        """Предсказание холста"""
        self.ui.export_to_png('image.png')  # Картинка холста 504 на 504

        # Манипуляции с изображением холста
        with Image.open("image.png") as image:
            # Сжимаем изображение с 784 на 784 до 28 * 28
            image.thumbnail(size=(28, 28))

            # Возвращаем изображение в режиме "L", где будем брать только альфа-канал, то есть прозрачность
            image = image.getchannel(channel="A")

        # Список значений альфа-канала всех 784 пикселей изображения
        pixels = []
        for y in range(28):
            for x in range(28):
                # Индексы от 0 до 27 из-за размера 28 * 28 и индексирования с нуля
                pixels.append(image.getpixel(xy=(x, y)))

        # Удалить файл картинки
        path = os.path.join(os.path.abspath(os.path.dirname(__file__)), 'image.png')
        os.remove(path)

        # 2-мерный numpy массив
        data = np.array(pixels, ndmin=2)
        
        
        def converter(num: float) -> int:
            s = str(num)
            if s.count("e") == 1:
                return 0
            
            s = s[2:5]
            
            if int(s[2]) > 4 and s[1] != "9":  # n % 10
                s = s[0] + str(int(s[1]) + 1) 
            elif s[0] != "9" and s[1] == "9":
                s = str(int(s[0]) + 1) + "0"
            else:
                s = s[:-1]
            
            return int(s)
                
        
        for i, el in enumerate(self.probabilities.children[-3::-3]):
            el.text = f"{converter(float(self.predictor.predict_proba(data)[0][i]))}%"
            # print(i, f"{converter(float(self.predictor.predict_proba(data)[0][i]))}%")
        
        # Вероятности на каждую из 10 цифр
        # Элементы в списке расположены в обратном поряждке, Label и Progressbar чередуются
        for i, el in enumerate(self.probabilities.children[-2::-3]):
            el.value = float(self.predictor.predict_proba(data)[0][i])

        # Отображаем конечное предсказание нейросети
        self.prediction_lbl.text = str(self.predictor.predict(data)[0])

    def predict(self, instance):
        print("predict")
        
        for i, el in enumerate(self.probabilities.children[-3::-3]):
            print(i, el.text)
            # print(float(self.predictor.predict_proba(data)[0][i]))
    
    def save_image(self, instance):
        print("save")

    def make_screenshot(self, instance):
        """Скриншот окна программы, если хочется запомнить результат"""
        self.export_to_png('i.png')


class MainApp(MDApp):
    """Основной класс интерфейса программы
    """

    def build(self):
        # get_color_from_hex(colors["Orange"]["900"])
        self.theme_cls.primary_palette = "Orange"
        self.theme_cls.primary_hue = '500'
        self.theme_cls.theme_style = 'Dark'  # Тёмная тема

        screen_manager = ScreenManager()
        screen_manager.add_widget(MainScreen(name="main_screen"))

        return screen_manager


if __name__ == '__main__':
    MainApp().run()
