"""Интерфейс приложения, осуществлённый при помощи библиотеки kivy.

https://kivy.org/doc/stable/api-kivy.graphics.html#module-kivy.graphics

Отсюда взята рисовалка - https://pythonprogramming.net/kivy-drawing-application-tutorial/
"""
from kivy.app import App
from kivy.core.window import Window

from kivy.uix.widget import Widget
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.progressbar import ProgressBar

from kivy.graphics import Line


import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
import pickle


Window.size = (784, 784)  # Размеры окна приложения


class Ui(Widget):
    def on_touch_down(self, touch):
        # print(touch)
        with self.canvas:
            touch.ud["line"] = Line(
                points=(touch.x, touch.y),
                width=10,
            )

    def on_touch_move(self, touch):
        # print(touch)
        touch.ud["line"].points += (touch.x, touch.y)

    def on_touch_up(self, touch):
        """Метод, срабатывающий после отжимания пальца или мыши. В данном случае можно сразу предсказывать число после рисовки, чтобы не нажимать на кнопку.
        """
        # with Image.open('image.png') as image:  # В теории можно сразу открывать в "L"
        #     # print(image.mode)  # RGBA

        #     # Преобразовываем изображение от 784 на 784 в 28 * 28
        #     image.thumbnail(size=(28, 28))
        #     image.save("image.png")

        #     # Возвращаем изображение в режиме "L", где будем брать только альфа-канал, то есть прозрачность
        #     image = image.getchannel(channel="A")

        # # Список значений альфа-канала всех 784 пикселей изобраэения
        # pixels = []
        # for y in range(28):
        #     for x in range(28):
        #         # Индексы от 0 до 27 из-за размера 28 * 28 и индексирования с нуля
        #         pixels.append(image.getpixel(xy=(x, y)))

        # # 2-мерный numpy массив
        # data = np.array(pixels, ndmin=2)

        # # Загружаем обученную модель
        # model: "MLPClassifier" = pickle.load(open("models/model.sav", "rb"))

        # # print(model.predict(data))
        # # Вероятности на каждую из 10 цифр
        # for index, probability in enumerate(model.predict_proba(data)[0]):
        #     print(index, probability)


class UiApp(App):
    def build(self):
        parent = Widget()

        self.ui = Ui()
        self.ui.size_hint = (0.8, 1.0)

        # Кнопка, по нажатию на которую холст будет очищаться
        clear_btn = Button(
            text="Clear",
            on_release=self.clear_canvas,
            size=(100, 100)
        )

        # Кнопка, по нажатию на которую нейронка будет давать предсказание
        predict_btn = Button(
            text="Save",  # "Predict"
            on_release=self.predict_canvas,
            size=(100, 100),
            pos=(100, 0)
        )

        parent.add_widget(self.ui)
        parent.add_widget(clear_btn)
        parent.add_widget(predict_btn)

        # pb = ProgressBar(max=100)
        # parent.add_widget(pb)

        return parent

    def clear_canvas(self, instance):
        """Очистка холста"""
        self.ui.canvas.clear()

    def predict_canvas(self, instance):
        """Предсказание холста
        """
        self.ui.size = (Window.size[0], Window.size[1])
        self.ui.export_to_png('image.png')


if __name__ == '__main__':
    UiApp().run()
