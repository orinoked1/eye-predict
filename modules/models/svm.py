from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import KFold
from sklearn import svm
import numpy


class SVM:
    def __init__(self, seed, df, run_name):
        # fix random seed for reproducibility
        numpy.random.seed(seed)
        self.df=df
        self.model = None
        self.history = None
        self.run_name = run_name
        self.seed = seed
        self.batch_size = 16
        self.num_epochs = 1
        self.max_review_length = 1500
        self.X = numpy.asanyarray(df.scanpath)
        self.y = numpy.asanyarray(df.binary_bid)
        self.X = sequence.pad_sequences(self.X, maxlen=self.max_review_length)

        dataset_size = len(self.X)
        self.X = self.X.reshape(dataset_size, -1)


    def run_model(self):
        #cross validation correct lables
        scores = []
        kf = KFold(n_splits=10, random_state=None, shuffle=True)
        for train_index, test_index in kf.split(self.X):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]
            # clf = tree()
            clf = svm.SVC(kernel="rbf", gamma=0.0000001)
            clf.fit(X_train, y_train)
            print("Train score: ", clf.score(X_train, y_train))
            print("Test score: ", clf.score(X_test, y_test))
            scores.append(clf.score(X_test, y_test))

        print("ACC: ", (sum(scores) / len(scores))*100)
