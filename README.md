# Predicting Drawn Digits

Предсказывание нарисованных цифр. Приложение реализовано с помощью фреймворка [kivy](https://kivy.org/) для создания графического интерфейса и библиотеки [scikit-learn](https://scikit-learn.org/stable/) для машинного обучения. Модели нейросетей обучаются на базе данных [MNIST](<https://ru.wikipedia.org/wiki/MNIST_(%D0%B1%D0%B0%D0%B7%D0%B0_%D0%B4%D0%B0%D0%BD%D0%BD%D1%8B%D1%85)>).

![Гифка](https://media.giphy.com/media/g05GUQPEqQzbjYGAbJ/giphy.gif)

<!-- ![Гифка](https://media.giphy.com/media/vFKqnCdLPNOKc/giphy.gif) -->

## Использование

1. Клонировать репозиторий:

   ```sh
   git clone https://github.com/PodYapolskiy/predicting-drawn-digits.git
   cd predicting-drawn-digits
   ```

2. Установить зависимости:

   ```sh
   pip install -r requirements.txt
   ```

3. Запустить программу:

   ```properties
   python main.py
   ```

## Создание своих моделей

1. Создать папку `datasets` и поместить туда CSV файл, который можно скачать, нажав [сюда](https://www.openml.org/data/get_csv/52667/mnist_784.arff]).
2. Запустить скрип (с аргументами или вводом):
   ```sh
   python model.py
   ```
