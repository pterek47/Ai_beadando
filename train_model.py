import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import pickle
import os

from sklearn.svm import SVC


def create_test_data():
    data = pd.read_csv('tweet_emotions.csv')
    texts = data['content'].fillna('')
    labels = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    with open('test_data.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)

    return X_test, y_test

def train_multinomial_nb():
    data = pd.read_csv('tweet_emotions.csv')
    texts = data['content'].fillna('')
    labels = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

    model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=5000), MultinomialNB())
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(f'Teszt pontszám: {test_score:.5f}')

    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("Konfúziós mátrix:", cm)

    with open('MultinomialNB_model.pkl', 'wb') as f:
        pickle.dump(model, f)

    print("A Multinomial modell sikeresen mentve!")
def train_svm():

    data = pd.read_csv('tweet_emotions.csv')
    texts = data['content'].fillna('')
    labels = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)


    model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=5000),
                          SVC(kernel='linear', probability=True))
    model.fit(X_train, y_train)


    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    print(f'Teszt pontszám: {test_score:.5f}')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    print("Konfúziós mátrix:", cm)

    with open('SVM_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    print("Az SVM modell sikeresen mentveasd!")
if __name__ == "__main__":
    if not os.path.exists('test_data.pkl'):
        print("A test_data.pkl fájl nem létezik, létrehozom...")
        create_test_data()

    train_multinomial_nb()
    train_svm()