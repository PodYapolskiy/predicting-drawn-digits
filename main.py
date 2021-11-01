"""Основной файл проекта

Суть программы: пользователь может рисовать цифры в разрешением 28 * 28, а нейросеть должна предсказать, какое число нарисовано.

RoadMap:
    Интерфейс, где пользователь может рисовать и  последствии это переведётся в картинку 28 * 28 и переведётся в numpy array.
    
    Неросеть: установка весов, чтобы каждый раз не обучать заново
    https://www.machinelearningmastery.ru/save-load-machine-learning-models-python-scikit-learn/

    Присоединение функционала к оболочке и окончание.
"""
# import nn
# import ui
import numpy as np
from PIL import Image


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
    with Image.open('image.png') as image:  # В теории можно сразу открывать в "L"
        # print(image.mode)  # RGBA

        # Преобразовываем изображение от 784 на 784 в 28 * 28
        image.thumbnail(size=(28, 28))
        image.save("image.png")

        # Возвращаем изображение в режиме "L", где будем брать только альфа-канал, то есть прозрачность
        image = image.getchannel(channel="A")
        # image.save('i.png')

    # print(image.getpixel(xy=(0, 0)))  # Индексы от 0 до 27 из-за размера 28 * 28 и индексирования с нуля
    # print(image)

    # Список
    pixels = []
    for y in range(28):
        for x in range(28):
            pixels.append(image.getpixel(xy=(x, y)))
    
    data = np.array(pixels, ndmin=2)

    from nn import make_prediction
    print(make_prediction(data))
    # visualize_first()
            

if __name__ == "__main__":
    main()
