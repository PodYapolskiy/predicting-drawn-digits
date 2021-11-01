"""Основной файл проекта

Суть программы: пользователь может рисовать цифры в разрешением 28 * 28, а нейросеть должна предсказать, какое число нарисовано.

RoadMap:
    Интерфейс, где пользователь может рисовать и  последствии это переведётся в картинку 28 * 28 и переведётся в numpy array.
    
    Неросеть: установка весов, чтобы каждый раз не обучать заново
    https://www.machinelearningmastery.ru/save-load-machine-learning-models-python-scikit-learn/

    Присоединение функционала к оболочке и окончание.
"""
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
import pickle

from ui import UiApp


def visualize_first():
    from nn import first_picture

    im = Image.new("L", (28, 28))

    i = 0
    for y in range(28):
        for x in range(28):
            im.putpixel(xy=(x, y), value=int(first_picture[i]))
            i += 1

    im.show()


def main():
    UiApp().run()

    with Image.open('image.png') as image:  # В теории можно сразу открывать в "L"
        # print(image.mode)  # RGBA

        # Преобразовываем изображение от 784 на 784 в 28 * 28
        image.thumbnail(size=(28, 28))
        image.save("image.png")

        # Возвращаем изображение в режиме "L", где будем брать только альфа-канал, то есть прозрачность
        image = image.getchannel(channel="A")
        # image.save('image.png')

    # Список значений альфа-канала всех 784 пикселей изобраэения
    pixels = []
    for y in range(28):
        for x in range(28):
            # Индексы от 0 до 27 из-за размера 28 * 28 и индексирования с нуля
            pixels.append(image.getpixel(xy=(x, y)))

    # 2-мерный numpy массив
    data = np.array(pixels, ndmin=2)

    # Загружаем обученную модель
    model: "MLPClassifier" = pickle.load(open("models/model.sav", "rb"))

    # print(model.predict(data))
    # Вероятности на каждую из 10 цифр
    for index, probability in enumerate(model.predict_proba(data)[0]):
        print(index, probability)

    # visualize_first()


if __name__ == "__main__":
    main()
