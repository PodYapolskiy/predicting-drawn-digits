"""Интерфейс приложения, осуществлённый при помощи библиотеки kivy.

https://kivy.org/doc/stable/api-kivy.graphics.html#module-kivy.graphics

Отсюда взята рисовалка - https://pythonprogramming.net/kivy-drawing-application-tutorial/
"""
from kivy.app import App
from kivy.uix.widget import Widget
from kivy.core.window import Window

from kivy.uix.button import Button

from kivy.graphics import Line


Window.size = (784, 784)  # Размеры окна приложения
"""
    1 28
    2 56
    3 84
    4 112
    5 140
    6 168
    7 196
    8 224
    9 252
    10 280
    11 308
    12 336
    13 364
    14 392
    15 420
    16 448
    17 476
    18 504
    19 532
    20 560
    21 588
    22 616
    23 644
    24 672
    25 700
    26 728
    27 756
    28 784
    29 812
    30 840
    31 868
    32 896
    33 924
    34 952
    35 980
    36 1008
    37 1036
    38 1064
    39 1092
"""

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
        # print("RELEASED!", touch)
        pass


class UiApp(App):
    def build(self):
        parent = Widget()
        self.ui = Ui()

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
