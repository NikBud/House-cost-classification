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

    # Import the dataframe with our data, place the labels in Y, and all other columns in X
    df = pd.read_csv("../Data/house_class.csv")
    X = df.iloc[:, 1:]
    Y = df.loc[:, "Price"]

    # Split the data into training and test sets (30% for testing, parameter - test_size),
    # Thanks to stratify=X['Zip_loc'].values, we split the data evenly
    # so that in the training set we see all possible values in the Zip_loc column.
    # The same proportional ratio (70 to 30) is maintained.
    # For example, if the value AE appears 10 times in the entire dataframe in the Zip_loc column,
    # the stratify parameter ensures it appears 7 times in the training set and 3 times in the test set.
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, stratify=X['Zip_loc'].values,
                                                        random_state=1)

    # Create a list of all encoders to be tested
    encoders = [
        OneHotEncoder(),
        OrdinalEncoder(),
        TargetEncoder(cols=['Zip_area', 'Zip_loc', 'Room'])
    ]

    # Create a list of 3 unique instances of the model. Create 3 identical and separate ones for experimental quality.
    models = [
        DecisionTreeClassifier(criterion="entropy", max_features=3, splitter="best", max_depth=6,
                               min_samples_split=4,
                               random_state=3)
    ] * 3

    # Create a list of encoder names to display the response in the format specified in the task
    encoder_names = [
        "OneHotEncoder",
        "OrdinalEncoder",
        "TargetEncoder"
    ]

    # Run a loop where we train the model for each encoder
    # (OneHotEncoder, OrdinalEncoder, TargetEncoder)
    # Testing data with different encoders helps find the optimal solution for our specific case.
    # Encoders are used to transform categorical data into numerical data.
    # Each of the 3 encoders has its own algorithm for transforming categorical data.
    # As we can see, OrdinalEncoder performs best.
    for encoder, model, encoder_name in zip(encoders, models, encoder_names):
        X_train_final, X_test_final = return_final_train_and_test_set(encoder, X_train, X_test, Y_train)
        model: DecisionTreeClassifier = model.fit(X_train_final, Y_train)
        predicted_Y_test = model.predict(X_test_final)

        # precision_recall_fscore_support method for obtaining statistics on the performance of our model
        # Accuracy_score alone is often not sufficient.
        # Detailed statistics help us improve the model and understand its weaknesses.
        # Precision indicates how well the model can distinguish a specific class from other classes.
        # Formula for precision = TP / (TP + FP)
        # Recall indicates how well the algorithm can identify (find, recognize) a specific class.
        # Formula for recall = TP / (TP + FN)
        # F-measure concisely combines both of the above metrics (precision, recall) into a single metric.
        # Support shows the number of occurrences of each class in the correct test labels.
        precision, recall, f1_score, support = precision_recall_fscore_support(Y_test, predicted_Y_test)
        print(encoder_name + ":" + str(round(f1_score.mean(), 2)))
