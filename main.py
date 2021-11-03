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
from kivy.app import App
from kivy.core.window import Window
from kivy.graphics import Line

from kivy.uix.label import Label
from kivy.uix.widget import Widget
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar
from kivy.uix.screenmanager import Screen, ScreenManager

import os
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
import pickle


Window.size = (WIDTH, HEIGHT) = (1000, 884)  # Размеры окна приложения


class Canvas(Widget):
    """Холст, область 784 на 784, расположенная в позиции (0, 100), в которой можно рисовать"""

    def __init__(self, predict_func, **kwargs):
        super().__init__(**kwargs)

        # Меняем размеры и положения холста относительно родителя
        self.size_hint = (784 / WIDTH, 784 / HEIGHT)
        self.pos_hint = {'x': 0, 'y': 100 / HEIGHT}

        # Назначаем функцию предсказания
        self.predict = predict_func

    def on_touch_down(self, touch):
        """Метод, срабатывающий при косании по холсту"""
        # Проверяем входит ли точка касания в область холста
        if not self.collide_point(*touch.pos):  # touch.pos = [touch.x, touch.y]
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


class Probabilities(GridLayout):
    """Элемент, в котором отображаются предсказания вероятности того или иного числа нейросетью"""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # Задаём размеры и позици, относительные внутри элемента родителя
        self.size_hint = (200 / WIDTH, 584 / HEIGHT)
        self.pos_hint = {'x': 784 / WIDTH, 'y': 100 / HEIGHT}

        # Добавляем ряды в grid layout
        for num in range(10):
            self.add_widget(Label(text=f"{num}", bold=True, size_hint_x=(40 / 216)))
            self.add_widget(ProgressBar(max=1))


class MainScreen(Screen):
    """Главный экран программы"""

    def __init__(self, **kw):
        super().__init__(**kw)

        # Добавляем холст, с переданной функцией предсказания
        self.ui = Canvas(predict_func=self.predict_canvas)
        self.add_widget(self.ui)

        # Добавляем кнопку очищения
        clear_btn = Button(
            text="Clear",
            on_release=self.clear_canvas,
            size_hint=(392 / WIDTH, 100 / HEIGHT)
        )
        self.add_widget(clear_btn)

        # Добавляем кнопку предсказания
        screenshot_btn = Button(
            text="Screenshot",
            on_release=self.make_screenshot,
            size_hint=(392 / WIDTH, 100 / HEIGHT),
            pos_hint={'x': 392 / WIDTH, 'y': 0}
        )
        self.add_widget(screenshot_btn)

        # Добавляем панель вероятностей
        self.probabilities = Probabilities(cols=2)
        self.add_widget(self.probabilities)

        # Добавляем слой с отображением конечного предказания нейросети
        self.prediction_lbl = Label(
            text="?",
            font_size='20sp',
            bold=True,
            size_hint=(216 / WIDTH, 100 / HEIGHT),
            pos_hint={'x': 784 / WIDTH}
        )
        self.add_widget(self.prediction_lbl)

        # Загружаем обученную нейросеть
        self.predictor: "MLPClassifier" = pickle.load(open("models/model a(0.976) hl(100_100_50) i(175).sav", "rb"))

    def clear_canvas(self, instance):
        """Очистка холста, вероятностей и конечного предсказания"""
        self.ui.canvas.clear()

        for pb in self.probabilities.children[-2::-2]:
            pb.value = 0

        self.prediction_lbl.text = ""

    def predict_canvas(self, instance):
        """Предсказание холста
        """
        self.ui.export_to_png('image.png')

        # Манипуляции с изображением холста
        with Image.open("image.png") as image:
            # Обрезаем холст с 1000 * 884 в 784 * 784
            image = image.crop(box=(0, 100, 784, 884))

            # Сжимаем изображение с 784 на 784 до 28 * 28
            image.thumbnail(size=(28, 28))

            # Возвращаем изображение в режиме "L", где будем брать только альфа-канал, то есть прозрачность
            image = image.getchannel(channel="A")
            # image.save("i.png")

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

        # Вероятности на каждую из 10 цифр
        # Элементы в списке расположены в обратном поряждке, Label и Progressbar чередуются
        for i, el in enumerate(self.probabilities.children[-2::-2]):
            el.value = float(self.predictor.predict_proba(data)[0][i])

        # Отображаем конечное предсказание нейросети
        self.prediction_lbl.text = str(self.predictor.predict(data)[0])

    def make_screenshot(self, instance):
        """Скриншот окна программы, если хочется запомнить результат"""
        self.export_to_png('i.png')


class MainApp(App):
    """Основной класс интерфейса программы
    """

    def build(self):
        screen_manager = ScreenManager()
        screen_manager.add_widget(MainScreen(name="main_screen"))

        return screen_manager


if __name__ == '__main__':
    MainApp().run()
