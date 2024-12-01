import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
import os
from sklearn.svm import SVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

def create_test_data():
    data = pd.read_csv('tweet_emotions.csv')
    texts = data['content'].fillna('')
    labels = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    with open('test_data.pkl', 'wb') as f:
        pickle.dump((X_test, y_test), f)
    return X_test, y_test
def train_kmeans():
    data = pd.read_csv('tweet_emotions.csv')
    texts = data['content'].fillna('')
    vectorizer = TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 3), max_features=5000)
    X = vectorizer.fit_transform(texts)
    num_clusters = 13
    model = KMeans(n_clusters=num_clusters, random_state=42)
    model.fit(X)
    with open('kmeans_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('vectorizer.pkl', 'wb') as f:
        pickle.dump(vectorizer, f)
    #print("K-Means model saved successfully!")
def train_multinomial_nb():
    data = pd.read_csv('tweet_emotions.csv')
    texts = data['content'].fillna('')
    labels = data['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)
    model = make_pipeline(TfidfVectorizer(lowercase=True, stop_words='english', ngram_range=(1, 2), max_features=5000), MultinomialNB())
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    test_score = model.score(X_test, y_test)
    #print(f'Teszt pontszám: {test_score:.5f}')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    #print("Konfúziós mátrix:", cm)
    with open('MultinomialNB_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    #print("A Multinomial modell sikeresen mentve!")
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
    #print(f'Teszt pontszám: {test_score:.5f}')
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    #print("Konfúziós mátrix:", cm)
    with open('svm_model.pkl', 'wb') as f:
        pickle.dump(model, f)
    #print("Az SVM modell sikeresen mentveasd!")
if __name__ == "__main__":
    if not os.path.exists('test_data.pkl'):
        print("A test_data.pkl fájl nem létezik, létrehozom...")
        create_test_data()
    train_multinomial_nb()
    #train_svm()
    #train_kmeans()
    #tanítás tesztelése: hozzá kell adni egy szöveg érzelem párt és azt másolni párszor. akkor szépen felismeri, mivel találkozott vele elég alkalommal.
    #pl:
    #git commit hej eloszor neutral
    #tweet_emotions.csv második sorába ezeket beillesztjük
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #47,anger,git commit hej
    #majd beillesztés után meg kell nyomni a fejlessze az ai unkat gombot. ez újra traineli a multinomialNB modelt.
    #Tesztelésre alkalmas szöveg:
#I absolutely love the smell of fresh flowers.
# I feel so guilty for forgetting her birthday.
# The movie was so disappointing; I expected better.
# I’m thrilled to be going on vacation next week.
# I feel a deep sense of nostalgia when I visit my childhood home.
# The thought of losing my job fills me with anxiety.
# I’m so proud of how much I’ve achieved this year.
# I can’t believe I embarrassed myself in front of everyone.
# The relief I felt when the test was over was immense.
# I feel a lot of jealousy when I see my friends getting promotions.
# Grief is overwhelming when you lose someone close.
# I was ecstatic to hear about the surprise party they planned for me.
# I regret not taking the opportunity when it was offered.
# I feel indifferent about the new movie that just came out.
# I love watching the sunset; it brings me peace.
# I’ve been feeling so bored lately with nothing to do.
# The surprise gift made me incredibly happy.
# I can't stop laughing at the silly joke he made.
# I hate the way she talks behind my back.
# I was shocked when I heard the news about the accident.