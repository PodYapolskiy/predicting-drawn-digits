"""Нейросеть, определяющая рукописное число.

Keyword arguments:
argument -- description
Return: return_description
"""
import pandas as pd
from sklearn.neural_network import MLPClassifier
import pickle


model: "MLPClassifier" = pickle.load(open("models/model.sav", "rb"))


def main():
    df = pd.read_csv('datasets/mnist_784.csv')

    x = df.loc[:, "pixel1":"pixel784"].values  # Все картинки со значениями пикселей от 0 до 256
    y = df["class"].values  # Все правильные ответы, которые должна предсказать нейронка

    assert len(x[0]) == 28 * 28
    assert x.shape == (70000, 784)
    print()
    first_picture = x[0]
    # print(first_picture)

    # Распределяем тренировочную и тестовую выборку
    x_train, x_test = x[:60000], x[60000:]
    y_train, y_test = y[:60000], y[60000:]

    # Наша модель - 3-слойный перцептрон
    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), verbose=True)
    mlp.fit(x_train, y_train)

    pickle.dump(mlp, open("models/model_1.sav", "wb"))

    print(mlp.predict([x[0]]))
    print(mlp.score(x_test, y_test))
    # model: "MLPClassifier" = pickle.load(open("models/model.sav", "rb"))
    # print(model.predict([x[0]]))


# Если запустить файл на прямую будет происходить весь процесс формирования нейронной сети от загрузки данных до её обучения 
if __name__ == "__main__":
    main()