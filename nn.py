"""Нейросеть, определяющая рукописное число.

Keyword arguments:
argument -- description
Return: return_description
"""
#%%
import pandas as pd
from sklearn.neural_network import MLPClassifier

'''
Сохранение весов:

    import h5py
    https://blog.stroganov.pro/%D1%81%D0%BE%D1%85%D1%80%D0%B0%D0%BD%D0%B5%D0%BD%D0%B8%D0%B5-%D0%B2%D0%B5%D1%81%D0%BE%D0%B2-%D0%BD%D0%B5%D0%B9%D1%80%D0%BE%D1%81%D0%B5%D1%82%D0%B8/
'''
df = pd.read_csv('datasets/mnist_784.csv')

x = df.loc[:, "pixel1":"pixel784"].values  # Все картинки со значениями пикселей от 0 до 256
y = df["class"].values  # Все правильные ответы, которые должна предсказать нейронка

assert len(x[0]) == 28 * 28
# print("\n" * 3)
# print(y[0])
# print("\n" * 3)
# print(df)
assert x.shape == (70000, 784)
print()
first_picture = x[0]
# print(first_picture)

# Распределяем тренировочную и тестовую выборку
x_train, x_test = x[:60000], x[60000:]
y_train, y_test = y[:60000], y[60000:]


def make_prediction(x_array):
    return mlp.predict(x_array)


#%%
# Наша модель - 3-слойный перцептрон
mlp = MLPClassifier(hidden_layer_sizes=(16, 16))
mlp.fit(x_train, y_train)

print(mlp.predict([x[0]]))
print(mlp.score(x_test, y_test))
