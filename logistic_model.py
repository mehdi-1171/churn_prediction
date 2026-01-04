from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess_data import PrepareData


class LogisticRegressionModel:

    def __init__(self, threshold):
        self.C = 1.0
        self.max_iter = 1000
        self.threshold = threshold
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.fitted = False
        self.y_predicted = None
        self.y_prob = None

    def get_data(self):
        data_instance = PrepareData(test_size=0.2, scale=True)
        self.X_train, self.X_test, self.y_train, self.y_test = data_instance.data_getter()

    def create_model(self):
        self.model = LogisticRegression(C=self.C, max_iter=self.max_iter)

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_predicted = self.model.predict(self.X_test)

    def predict_proba(self):
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

    def evaluate(self):
        # Change with Threshold:
        self.y_predicted = (self.y_prob >= self.threshold).astype(int)
        print("Classification Report:")
        print(classification_report(self.y_test, self.y_predicted))
        print("ROC-AUC Score:", roc_auc_score(self.y_test, self.y_prob))

        cm = confusion_matrix(self.y_test, self.y_predicted)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

    def process_handler(self):
        self.get_data()
        self.create_model()
        self.train()
        self.predict()
        self.predict_proba()
        self.evaluate()


""" TEST """
# log = LogisticRegressionModel(threshold=0.3)
# log.get_data()
# log.create_model()
# log.train()
# log.predict()
# log.predict_proba()
# log.evaluate()
