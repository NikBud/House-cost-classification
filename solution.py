import os
import requests
import sys
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from category_encoders import TargetEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import precision_recall_fscore_support


def return_final_train_and_test_set(enc, X_train, X_test, Y_train):
    X_train_final = None
    X_test_final = None
    if isinstance(enc, TargetEncoder):
        X_train_final = pd.DataFrame(enc.fit_transform(X=X_train, y=Y_train,
                                                       index=X_train.index))
        X_test_final = pd.DataFrame(enc.transform(X=X_test),
                                    index=X_test.index)

    else:
        X_train_transformed = None
        X_test_transformed = None
        if isinstance(enc, OrdinalEncoder):
            X_train_transformed = pd.DataFrame(enc.fit_transform(X=X_train[['Zip_area', 'Zip_loc', 'Room']]),
                                               index=X_train.index)
            X_test_transformed = pd.DataFrame(enc.transform(X=X_test[['Zip_area', 'Zip_loc', 'Room']]),
                                              index=X_test.index)

        elif isinstance(enc, OneHotEncoder):
            X_train_transformed = pd.DataFrame(enc.fit_transform(X=X_train[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                               index=X_train.index)
            X_test_transformed = pd.DataFrame(enc.transform(X=X_test[['Zip_area', 'Zip_loc', 'Room']]).toarray(),
                                              index=X_test.index)

        if X_train_transformed is None or X_test_transformed is None:
            return None, None

        X_train_final = X_train[['Area', 'Lon', 'Lat']].join(X_train_transformed)
        X_train_final.columns = X_train_final.columns.astype(str)
        X_test_final = X_test[['Area', 'Lon', 'Lat']].join(X_test_transformed)
        X_test_final.columns = X_test_final.columns.astype(str)

    return X_train_final, X_test_final


def download_data():
    if not os.path.exists('../Data'):
        os.mkdir('../Data')
    if 'house_class.csv' not in os.listdir('../Data'):
        sys.stderr.write("[INFO] Dataset is loading.\n")
        url = "https://www.dropbox.com/s/7vjkrlggmvr5bc1/house_class.csv?dl=1"
        r = requests.get(url, allow_redirects=True)
        open('../Data/house_class.csv', 'wb').write(r.content)
        sys.stderr.write("[INFO] Loaded.\n")


if __name__ == '__main__':
    # Download data if it is unavailable.
    download_data()

    # Испортируем датафрейм с нашими данными, помещаем в У наши метки, а в Х - все остальные столбцы
    df = pd.read_csv("../Data/house_class.csv")
    X = df.iloc[:, 1:]
    Y = df.loc[:, "Price"]

    # Делим данные на тренировочноые и тестовые (30% - тестовых, параметр - test_size),
    # при этом благодаря: stratify = X['Zip_loc'].values, мы делим данные равномерно,
    # так чтоб в обучающем наборе мы видели все возможные значения в столбце Zip_loc.
    # При этом в таком же пропорциональном соотношении (70 на 30)
    # Например встречается во всем датафрейме в столбце Zip_loc значение АЕ - 10 раз,
    # Параметр stratify сделает так, чтоб в обучающем наборе оно встречалось 7 раз, а в тестовом - 3
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=X['Zip_loc'].values,
                                                        random_state=1)

    # Создаем список со всеми преобразователями, которые мы будем тестировать
    encoders = [
        OneHotEncoder(),
        OrdinalEncoder(),
        TargetEncoder(cols=['Zip_area', 'Zip_loc', 'Room'])
    ]

    # Создаем список с 3 уникальными экземплярами модели. Создаем 3 одинаковые и отдельные для качества эксперемента.
    models = [
        DecisionTreeClassifier(criterion="entropy", max_features=3, splitter="best", max_depth=6,
                               min_samples_split=4,
                               random_state=3)
    ] * 3

    # Создаем список со всеми имена преобразователей, чтоб отобразить ответ в том виде, как сказано в ТЗ
    encoder_names = [
        "OneHotEncoder",
        "OrdinalEncoder",
        "TargetEncoder"
    ]

    # Запускаем цикл, в котором мы обучим модель для каждого преобразователя
    # (OneHotEncoder, OrdinalEncoder, TargetEncoder)
    # Тестирование данных с разными преобразователями помогает найти оптимальное решение конкретно в нашем случае
    # Преобразователи используются для трансформации категориальных данных в числовые
    # Каждый из 3 преобразователей имеет свой алгоритм трансформации категориальных данных
    # Как мы видим лучше всего справляется OrdinalEncoder
    for encoder, model, encoder_name in zip(encoders, models, encoder_names):
        X_train_final, X_test_final = return_final_train_and_test_set(encoder, X_train, X_test, Y_train)
        model: DecisionTreeClassifier = model.fit(X_train_final, Y_train)
        predicted_Y_test = model.predict(X_test_final)

        # precision_recall_fscore_support метод для получения статистики по работе нашей модели
        # Зачастую не достаточно лишь accurancy_score
        # Благодаря подробной статистике мы можем улучшить модель, понять в чем ее слабые стороны
        # Precision(точность) показывает насколько хорошо мродель умеет отличать определенный класс от других классов
        # Формула точности = TP / (TP + FP)
        # Recall(полнота) показывает насколько правильно алгоритм умеет определять(находить, узнавать) определенный кл.
        # Формула точности = TP / (TP + FN)
        # F-measure лаконично объединяет в один показатель оба из показателей выше (precision, recall)
        # Support показывает количество появлений каждого класса в правильных тестовых метках
        precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, predicted_Y_test)
        print(encoder_name + ":" + str(round(f1_score.mean(), 2)))
