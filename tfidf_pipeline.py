import time
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
# Tokenizers
from nltk.tokenize import WhitespaceTokenizer
from nltk.tokenize import WordPunctTokenizer
# Vectorizers
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# Classifiers
from sklearn.svm import LinearSVC
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import Lasso
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Perceptron

seed = 895376


class Pipeline:

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.vectorizer = TfidfVectorizer(ngram_range=(1, 1), max_features=50000)
        # self.classifier = LinearSVC(random_state=seed)
        self.classifier = LogisticRegression(random_state=seed, multi_class='multinomial')
        # self.classifier = RidgeClassifier(random_state=seed)
        # self.classifier = KNeighborsClassifier(n_jobs=4, n_neighbors=1)
        # self.classifier = KNeighborsClassifier(n_jobs=4, n_neighbors=3)
        # self.classifier = KNeighborsClassifier(n_jobs=4, n_neighbors=5)
        self.classifier = Perceptron(random_state=seed)
        # Raw file
        self.train_file = "raw/train_tweets.txt"
        self.test_file = "raw/test_tweets_unlabeled.txt"
        # Cleaned file
        self.train_file_cleaned = "data/train_tweets_cleaned.txt"
        self.test_file_cleaned = "data/test_tweets_cleaned.txt"
        self.total_file_cleaned = "data/total_tweets_cleaned.txt"
        # Vector File
        self.train_vector = "vector/train.vec"
        self.test_vector = "vector/test.vec"
        # Label File
        self.train_label = "label/train_label.txt"
        self.test_label = "label/test_label.csv"

    def tokenize(self):
        print("Tokenizing...")
        train_file_cleaned = open(self.train_file_cleaned, 'w')
        test_file_cleaned = open(self.test_file_cleaned, 'w')
        total_file_cleaned = open(self.total_file_cleaned, 'w')
        train_label = open(self.train_label, 'w')
        with open(self.train_file) as train_data:
            for line in train_data:
                label, tweet = line.strip().split('\t', 1)[:2]
                train_label.write(label + '\n')
                tokenized_tweet = " ".join(self.tokenizer.tokenize(tweet))
                train_file_cleaned.write(tokenized_tweet + '\n')
                total_file_cleaned.write(tokenized_tweet + '\n')
        with open(self.test_file) as test_data:
            for line in test_data:
                tokenized_tweet = " ".join(self.tokenizer.tokenize(line))
                test_file_cleaned.write(tokenized_tweet + '\n')
                total_file_cleaned.write(tokenized_tweet + '\n')

    def vectorize(self):
        print("Fitting vectorizer...")
        self.vectorizer.fit(open(self.total_file_cleaned))
        print("Vectorizing train file...")
        train_vector = self.vectorizer.transform(open(self.train_file_cleaned))
        print("Train vector: ", train_vector.shape)
        print("Vectorizing test file...")
        test_vector = self.vectorizer.transform(open(self.test_file_cleaned))
        print("Test vector: ", test_vector.shape)
        print("Saving...")
        pickle.dump(train_vector, open(self.train_vector, 'wb'))
        pickle.dump(test_vector, open(self.test_vector, 'wb'))

    def evaluate(self):
        train_vector = pickle.load(open(self.train_vector, 'rb'))
        train_label = []
        with open(self.train_label) as file:
            for line in file:
                train_label.append(int(line))
        print("Total Data: ", train_vector.shape)
        X_train, X_evl, y_train, y_evl = train_test_split(train_vector, train_label, test_size=0.5, random_state=seed)
        _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
        _, X_evl, _, y_evl = train_test_split(X_evl, y_evl, test_size=0.1, random_state=seed)
        print("Training set has {} instances. Test set has {} instances.".format(X_train.shape[0], X_evl.shape[0]))
        start = time.time()
        print("Training Classifier...")
        self.classifier.fit(X_train, y_train)
        pred_labels = self.classifier.predict(X_evl)
        print("Training successfully in %s seconds " % int(time.time() - start))
        print("Evaluate Accuracy: %0.2f" % (accuracy_score(y_evl, pred_labels) * 100))

    def classify(self):
        train_vector = pickle.load(open(self.train_vector, 'rb'))
        train_label = []
        with open(self.train_label) as file:
            for line in file:
                train_label.append(int(line))
        print("Total Data: ", train_vector.shape)
        start = time.time()
        print("Training Classifier...")
        self.classifier.fit(train_vector, train_label)
        print("Training successfully in %s seconds " % int(time.time() - start))
        print("Predicting...")
        test_vector = pickle.load(open(self.test_vector, 'rb'))
        test_label = self.classifier.predict(test_vector)
        df = pd.DataFrame(test_label, columns=['Predicted'])
        df.index += 1
        df.index.name = 'Id'
        df.to_csv(self.test_label)


pipe = Pipeline()
pipe.tokenize()
pipe.vectorize()
pipe.evaluate()
# pipe.classify()
