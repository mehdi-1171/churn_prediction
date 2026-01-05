from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.model_selection import RandomizedSearchCV
from pandas import Series
import seaborn as sns
import matplotlib.pyplot as plt
import os

from preprocess_data import PrepareData, FeatureEngineer


class XGBoostModel:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    FIG_PATH = os.path.join(ROOT_DIR, 'fig/')

    def __init__(self, feature, threshold, param_dist):
        self.feature = feature
        self.param_dist = param_dist
        self.threshold = threshold
        self.n_estimators = 500,
        self.max_depth = 8,
        self.learning_rate = 0.01
        self.subsample = 0.7
        self.min_child_weight = 5
        self.col_sample_by_tree = 0.8
        self.objective = "binary:logistic"
        self.eval_metric = "auc"
        self.scale_pos_weight = 2.9
        self.random_state = 42
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None
        self.model = None
        self.y_prob = None
        self.y_predicted = None

    def get_data(self):
        if self.feature:
            data_instance = FeatureEngineer(test_size=0.2, scale=True)
        else:
            data_instance = PrepareData(test_size=0.2, scale=False)
        self.X_train, self.X_test, self.y_train, self.y_test = data_instance.data_getter()

    def find_param(self):
        if len(self.param_dist) > 0:
            xgb = XGBClassifier(
                objective="binary:logistic",
                eval_metric="auc",
                random_state=42,
                use_label_encoder=False
            )
            random_search = RandomizedSearchCV(
                estimator=xgb,
                param_distributions=self.param_dist,
                n_iter=30,
                scoring="roc_auc",
                cv=5,
                n_jobs=-1,
                verbose=1,
                random_state=42
            )
            random_search.fit(x.X_train, x.y_train)
            self.best_xgb = random_search.best_estimator_

            self.n_estimators = self.best_xgb.n_estimators,
            self.max_depth = self.best_xgb.max_depth,
            self.learning_rate = self.best_xgb.learning_rate
            self.subsample = self.best_xgb.subsample
            self.min_child_weight = self.best_xgb.min_child_weight
            self.col_sample_by_tree = self.best_xgb.colsample_bytree
            self.scale_pos_weight = self.best_xgb.scale_pos_weight

    def create_model(self):
        self.model = XGBClassifier(params={
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.col_sample_by_tree,
            "objective": self.objective,
            "eval_metric": self.eval_metric,
            "scale_pos_weight": self.scale_pos_weight,
            "random_state": self.random_state,
            "min_child_weight": self.min_child_weight
        })

    def train(self):
        self.model.fit(self.X_train, self.y_train)

    def predict_proba(self):
        self.y_prob = self.model.predict_proba(self.X_test)[:, 1]

    def predict(self):
        if self.threshold == 0:
            self.y_predicted = self.model.predict(self.X_test)
        else:
            self.y_predicted = (self.y_prob >= self.threshold).astype(int)

    def evaluate(self):
        self.predict_proba()
        self.predict()
        print(f"Classification Report (Threshold={self.threshold}):")
        print(classification_report(self.y_test, self.y_predicted))
        print("ROC-AUC Score:", roc_auc_score(self.y_test, self.y_prob))
        cm = confusion_matrix(self.y_test, self.y_predicted)
        plt.figure(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Oranges")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix - XGBoost")
        plt.savefig(self.FIG_PATH + 'XGBoost_heatmap.png')
        plt.show()

    def plot_feature_importance(self, top_n=15):
        importance = Series(
            self.model.feature_importances_,
            index=self.X_train.columns
        ).sort_values(ascending=False)

        plt.figure(figsize=(10, 6))
        importance.head(top_n).plot(kind="barh")
        plt.title("Top Feature Importances - XGBoost")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.subplots_adjust(left=0.35)
        plt.savefig(self.FIG_PATH + 'XGBoost_Importance.png')
        plt.show()

    def process_handler(self):
        self.get_data()
        self.find_param()
        self.create_model()
        self.train()
        self.evaluate()
        self.plot_feature_importance()


""" TEST """
param_dist = {
    "n_estimators": [300, 500, 600],
    "max_depth": [2, 3, 4, 8],
    "learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "subsample": [0.6, 0.6, 0.7, 0.8],
    "colsample_bytree": [0.7, 0.8, 0.9, 0.95, 1],
    "min_child_weight": [0, 0.5, 1, 3, 5, 10],
    "scale_pos_weight": [2.6, 2.8, 2.9, 3.2]
}
param2 = {}
x = XGBoostModel(feature=True, threshold=0, param_dist=param_dist)
x.process_handler()

