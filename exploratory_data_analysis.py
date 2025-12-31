from pandas import DataFrame, concat, read_csv
import os

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(ROOT_DIR, 'data/')
file_name = 'Telco Customer Churn.csv'

df = read_csv(DATA_PATH + file_name)
df.columns

""" EDA """
df.info()
# Data Quality Check
df.describe().T

class CleanData:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT_DIR, 'data/')
    FILE_NAME = 'Telco Customer Churn.csv'

    def __init__(self):
        self.df = DataFrame()

    def read_data(self):
        self.df = read_csv(self.DATA_PATH + self.FILE_NAME)

    def show_data(self):
        print(f'Data Info: {self.df.info()}')
        print(f'Data N Unique: {self.df.nunique()}')

    def check_data(self):
        if len(list(df.columns[df.isnull().any()]))>1:
            print('')
    def convert_gender(self):
        self.df['gender'] = self.df['gender'].replace({'Male': 1, 'Female': 0})

    def convert_object2boolean(self):
        cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        self.df[cols] = self.df[cols].replace({'Yes': 1, 'No': 0})



df['OnlineBackup'].unique()
