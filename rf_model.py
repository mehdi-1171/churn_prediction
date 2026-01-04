from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from pandas import Series
import seaborn as sns
import matplotlib.pyplot as plt
import os

from preprocess_data import PrepareData


class RandomForestModel:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FIG_PATH = os.path.join(ROOT_DIR, 'fig/')

    def __init__(self, threshold, auto_param):
        self.auto_param = auto_param
        self.threshold = threshold
        self.n_estimators = 500
        self.max_depth = None
        self.min_samples_split = 10
        self.min_samples_leaf = 20
        self.random_state = 42
        self.class_weight = {0: 1, 1: 2}
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.y_predicted = None
        self.y_prob = None

    def get_data(self):
        data_instance = PrepareData(test_size=0.2, scale=False)
        self.X_train, self.X_test, self.y_train, self.y_test = data_instance.data_getter()

    def tune_parameters(self):
        if self.auto_param:
            param_dist = {
                'n_estimators': [200, 300, 500],
                'max_depth': [4, 6, 8, 10, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 5, 10, 20],
                'max_features': ['sqrt', 'log2'],
                'class_weight': [
                    {0: 1, 1: 2},
                    {0: 1, 1: 3}
                ]
            }
            rf = RandomForestClassifier(random_state=42)
            random_search = RandomizedSearchCV(
                rf,
                param_distributions=param_dist,
                n_iter=30,
                scoring='roc_auc',
                cv=5,
                n_jobs=-1,
                random_state=42
            )
            random_search.fit(self.X_train, self.y_train)

            best_rf = random_search.best_estimator_
            print(f'Best Parameter: {best_rf}')

    def create_model(self):
        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
            class_weight=self.class_weight
        )

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict_proba(self):
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

    def predict(self):
        self.y_predicted = (self.y_prob >= self.threshold).astype(int)

    def evaluate(self):
        self.predict_proba()
        self.predict()

        print(f"Classification Report (Threshold={self.threshold}):")
        print(classification_report(self.y_test, self.y_predicted))
        print("ROC-AUC Score:", roc_auc_score(self.y_test, self.y_prob))

        cm = confusion_matrix(self.y_test, self.y_predicted)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Greens")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - Random Forest")
        plt.show()
        plt.savefig(self.FIG_PATH + 'RF_ConfusionMatrix.png')

    def plot_feature_importance(self, top_n=15):
        feature_importance = Series(
            self.model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        feature_importance.head(top_n).plot(kind='barh')

        plt.title("Top Feature Importances - Random Forest")
        plt.xlabel("Importance")

        plt.gca().invert_yaxis()

        plt.tight_layout()
        plt.subplots_adjust(left=0.35)

        plt.show()
        plt.savefig(self.FIG_PATH + 'RF_Importance.png')

    def process_handler(self):
        self.get_data()
        self.create_model()
        self.train()
        self.evaluate()
        self.plot_feature_importance()


""" TEST """
k = RandomForestModel(threshold=0.5, auto_param=False)
k.process_handler()

# best Practice:
# RandomForestClassifier(class_weight={0: 1, 1: 2}, min_samples_leaf=20,
#                        min_samples_split=10, n_estimators=500, random_state=42)