import pandas as pd 

from sklearn.feature_extraction.text import TfidfVectorizer # tfidf -> term frequency inverse document frequency, szavak fontossagat meri
from sklearn.model_selection import train_test_split # a dataset-et szetbontja train es test reszre
from sklearn.naive_bayes import MultinomialNB # Naive Bayes modell
from sklearn.pipeline import make_pipeline # pipeline, ami osszefuzi a tfidf-et es a modellt
import pickle # modell mentese 

data = pd.read_csv('tweet_emotions.csv') # kaggle-rol van a dataset

# adatok feldolgozasa, szetvalasztasa
texts = data['content'].fillna('')
labels = data['sentiment']
# adatok felosztasa tanulasra es tesztelesre
X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.2, random_state=42)

# Pipeline letrehozasa
model = make_pipeline(TfidfVectorizer(), MultinomialNB())
# Modell tanitasa
model.fit(X_train, y_train)

# Modell ertekelese a teszten
test_score = model.score(X_test, y_test)
print(f'Teszt pontszám: {test_score:.3f}')

# Modell mentese
with open('sentiment_model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("A modell sikeresen mentve!")