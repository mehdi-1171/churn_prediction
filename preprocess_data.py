from pandas import DataFrame, concat, get_dummies, to_numeric, crosstab, read_excel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from numpy import array
import os
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder


class PrepareData:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT_DIR, 'data/')
    FILE_NAME = 'churn_data_clean.xlsx'

    def __init__(self, test_size, scale):
        self.test_size = test_size
        self.scale = scale
        self.df = DataFrame()
        self.target = 'Churn'
        self.X = DataFrame()
        self.y = DataFrame()
        self.X_train, self.X_test, self.y_train, self.y_test = None, None, None, None

    def data_getter(self):
        self.preprocess_handler()
        return self.X_train, self.X_test, self.y_train, self.y_test

    def read_data(self):
        self.df = read_excel(self.DATA_PATH + self.FILE_NAME)

    @staticmethod
    def convert_digit(x):
        if (x == 'Yes') or (x == 'yes'):
            y = 1
        else:
            y = 0
        return y

    def convert_categorical_boolean_columns(self):
        cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup',
                'DeviceProtection', 'TechSupport',
                'StreamingTV', 'StreamingMovies']
        for c in cols:
            self.df[c] = self.df[c].apply(lambda x: self.convert_digit(x))

    def encode_categorical_data(self):
        cols = ['InternetService', 'Contract', 'PaymentMethod']
        self.df = get_dummies(self.df, columns=cols)
        dummy_cols = [c for c in self.df.columns if any(prefix in c for prefix in cols)]
        self.df[dummy_cols] = self.df[dummy_cols].astype(int)

    def clean_data(self):
        if 'customerID' in self.df.columns:
            self.df = self.df.drop(['customerID'], axis=1)

    def separate_depend_independent_parameter(self):
        self.X = self.df.drop(self.target, axis=1)
        self.y = self.df[self.target]

    def scale_data(self):
        if self.scale:
            scaler = MinMaxScaler()
            self.X = scaler.fit_transform(self.X)

    def separate_test_train(self):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=self.test_size, random_state=42, stratify=self.y
        )

    def preprocess_handler(self):
        self.read_data()
        self.convert_categorical_boolean_columns()
        self.encode_categorical_data()
        self.clean_data()
        self.separate_depend_independent_parameter()
        self.scale_data()
        self.separate_test_train()


""" TEST """
# p = PrepareData(test_size=0.2, scale=True)
# p.preprocess_handler()
#


class FeatureEngineer(PrepareData):

    def __init__(self, test_size, scale):
        super().__init__(test_size, scale)

    def add_tenure_based_features(self):
        self.df['tenure_years'] = self.df['tenure'] / 12
        self.df['is_new_customer'] = (self.df['tenure'] < 6).astype(int)
        self.df['is_loyal_customer'] = (self.df['tenure'] > 24).astype(int)

    def cost_business_feature(self):
        self.df['monthly_to_total_ratio'] = self.df['MonthlyCharges'] / (self.df['TotalCharges'] + 1)
        self.df['high_monthly_charge'] = (self.df['MonthlyCharges'] > self.df['MonthlyCharges'].median()).astype(int)

    def service_density(self):
        service_cols = [
            'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
            'TechSupport', 'StreamingTV', 'StreamingMovies'
        ]
        self.df['num_services'] = self.df[service_cols].sum(axis=1)

    def add_feature(self):
        self.add_tenure_based_features()
        self.cost_business_feature()
        self.service_density()

    def preprocess_handler(self):
        self.read_data()
        self.convert_categorical_boolean_columns()
        self.encode_categorical_data()
        self.clean_data()
        self.add_feature()
        self.separate_depend_independent_parameter()
        self.scale_data()
        self.separate_test_train()


""" TEST """
# f = FeatureEngineer(test_size=0.2, scale=True)
# f.read_data()
# f.convert_categorical_boolean_columns()
# f.encode_categorical_data()
# f.clean_data()
# f.add_feature()