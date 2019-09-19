import time
import pickle
import os.path
import numpy as np
from gensim.models import Word2Vec
from nltk.tokenize import WordPunctTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

seed = 895376
max_bytes = 2 ** 31 - 1


class Pipeline:

    def __init__(self):
        self.tokenizer = WordPunctTokenizer()
        self.classifier = LogisticRegression(random_state=seed, solver='lbfgs', multi_class='multinomial')
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
        # Model
        self.model = "model/model.bin"

    def tokenize(self):
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

    def train(self, dimension=64):
        data = []
        with open(self.total_file_cleaned) as file:
            for line in file:
                data.append(line.split(' '))
        start = time.time()
        print("Training Word2Vec model...")
        model = Word2Vec(data, size=dimension, seed=seed, min_count=1)
        model.save(self.model)
        print("Model saved in : %s seconds " % (time.time() - start))

    def vectorize(self, dimension=64):
        model = Word2Vec.load(self.model)
        train_vector, test_vector = [], []
        count = 0
        with open(self.train_file_cleaned) as file:
            for line in file:
                vector = [0 for _ in range(dimension)]
                words = 0
                for word in line.split(' '):
                    words += 1
                    vector += model[word]
                if words != 0:
                    vector /= words
                train_vector.append(vector)
                count += 1
                if count % 10000 == 0:
                    print("%s tweets vectorized..." % count)
        # write train data
        train_out = pickle.dumps(train_vector)
        with open(self.train_vector, 'wb') as f_out:
            for idx in range(0, len(train_out), max_bytes):
                f_out.write(train_out[idx:idx + max_bytes])
        with open(self.test_file_cleaned) as file:
            for line in file:
                vector = [0 for _ in range(dimension)]
                words = 0
                for word in line.split(' '):
                    words += 1
                    vector += model[word]
                if words != 0:
                    vector /= words
                test_vector.append(vector)
                count += 1
                if count % 10000 == 0:
                    print("%s tweets vectorized..." % count)
        # write test data
        test_out = pickle.dumps(test_vector)
        with open(self.test_vector, 'wb') as f_out:
            for idx in range(0, len(test_out), max_bytes):
                f_out.write(test_out[idx:idx + max_bytes])

    def evaluate(self):
        print("Reading data...")
        train_in = bytearray(0)
        input_size = os.path.getsize(self.train_vector)
        with open(self.train_vector, 'rb') as f_in:
            for _ in range(0, input_size, max_bytes):
                train_in += f_in.read(max_bytes)
        train_vector = pickle.loads(train_in)
        train_label = []
        with open(self.train_label) as file:
            for line in file:
                train_label.append(int(line))
        print("Total Data: (%s, %s)" % (len(train_vector), len(train_vector[0])))
        X_train, X_evl, y_train, y_evl = train_test_split(train_vector, train_label, test_size=0.5, random_state=seed)
        _, X_train, _, y_train = train_test_split(X_train, y_train, test_size=0.1, random_state=seed)
        _, X_evl, _, y_evl = train_test_split(X_evl, y_evl, test_size=0.1, random_state=seed)
        print("Training set has {} instances. Test set has {} instances.".format(len(X_train), len(X_evl)))
        start = time.time()
        print("Training Classifier...")
        self.classifier.fit(np.array(X_train), np.array(y_train))
        print("Training successfully in %s seconds " % int(time.time() - start))
        # print("Evaluating Classifier...")
        pred_labels = self.classifier.predict(X_evl)
        print("Evaluate Accuracy: %0.4f" % (accuracy_score(y_evl, pred_labels)))


pipe = Pipeline()
pipe.tokenize()
pipe.train(dimension=32)
pipe.vectorize(dimension=32)
pipe.evaluate()
