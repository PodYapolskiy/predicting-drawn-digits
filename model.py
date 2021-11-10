"""Нейросеть, определяющая рукописное число.
"""
import pandas as pd
from PIL import Image
from sklearn.neural_network import MLPClassifier
import pickle


def visualize_first(arr):
    """arr - массив с 784 значениями"""
    im = Image.new("L", (28, 28))

    i = 0
    for y in range(28):
        for x in range(28):
            im.putpixel(xy=(x, y), value=int(arr[i]))
            i += 1

    im.show()


def main():
    df = pd.read_csv('datasets/mnist_784.csv')

    # Все картинки со значениями пикселей от 0 до 256
    x = df.loc[:, "pixel1":"pixel784"].values
    # Все правильные ответы, которые должна предсказать нейронка
    y = df["class"].values

    assert len(x[0]) == 28 * 28
    assert x.shape == (70000, 784)

    # Распределяем тренировочную и тестовую выборку
    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    hidden_layers = [100, 100, 50]
    max_iter = 150

    # Наша модель - многослойный перцептрон
    mlp = MLPClassifier(
        hidden_layer_sizes=tuple(hidden_layers),
        max_iter=max_iter,
        verbose=True,
        # alpha=0.001
    )
    mlp.fit(x_train, y_train)

    print(mlp.predict([x[0]]))  # 5
    accuracy = mlp.score(x_test, y_test)
    print(accuracy)

    pickle.dump(
        mlp,
        open(
            f"models/model a({accuracy:.3}) hl({'_'.join([str(i) for i in hidden_layers])}) i({max_iter}).sav",
            "wb"
        )
    )


# Если запустить файл на прямую будет происходить весь процесс формирования нейронной сети от загрузки данных до её обучения
if __name__ == "__main__":
    main()
