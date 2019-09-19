from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline

seed = 895376
print("Load data")
data = []
with open("data/train_tweets_cleaned.txt") as file:
    for line in file:
        data.append(line)
label = []
with open("label/train_label.txt") as file:
    for line in file:
        label.append(int(line))
_, data, _, label = train_test_split(data, label, test_size=0.01, random_state=seed)
print("Init pipeline")
tfidf = TfidfVectorizer(stop_words='english', ngram_range=(1, 1), max_features=50000)
svm = LinearSVC(random_state=seed)
pipe = Pipeline(steps=[('vectorizer', tfidf), ('classifier', svm)])
param_grid = {
    # 'vectorizer__ngram_range': [(1, 1), (1, 2), (1, 3), (2, 3)],
    # 'vectorizer__max_features': [100, 1000, 10000, 100000],
    # 'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
    # 'classifier__penalty': ['l2', 'l1'],
    # 'classifier__loss': ['squared_hinge', 'hinge'],
    # 'classifier__dual': [True, False],
    'classifier__tol': [1e-5, 1e-4, 1e-3]
}
search = GridSearchCV(pipe, param_grid, scoring='accuracy', cv=2, n_jobs=4, error_score=0.0)
print("Fitting")
search.fit(data, label)
print("Best parameter (score=%0.3f):" % search.best_score_)
print(search.best_params_)
