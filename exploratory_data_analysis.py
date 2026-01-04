from pandas import DataFrame, concat, read_csv, to_numeric, crosstab
import os
import seaborn as sns
import matplotlib.pyplot as plt


class CleanData:
    def __init__(self, df):
        self.df = df

    def data_getter(self):
        self.clean_process()
        return self.df

    def show_data(self):
        print(f'Data Info: {self.df.info()}')
        print(f'Data N Unique: {self.df.nunique()}')

    def check_data(self):
        if len(list(self.df.columns[self.df.isnull().any()]))>1:
            print('Exist Null data')

    def convert_gender(self):
        self.df['gender'] = self.df['gender'].replace({'Male': 1, 'Female': 0})

    def convert_object2boolean(self):
        cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
        self.df[cols] = self.df[cols].replace({'Yes': 1, 'No': 0})

    def convert_type_data(self):
        self.df['TotalCharges'] = (
            to_numeric(self.df['TotalCharges'], errors='coerce')
            .fillna(0.0)
        )

    def clean_process(self):
        # self.show_data()
        self.check_data()
        self.convert_gender()
        self.convert_object2boolean()
        self.convert_type_data()


class ExploratoryAnalysis:
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PATH = os.path.join(ROOT_DIR, 'data/')
    FILE_NAME = 'Telco Customer Churn.csv'

    def __init__(self):
        self.df = DataFrame()

    def read_data(self):
        self.df = read_csv(self.DATA_PATH + self.FILE_NAME)

    def clean_data(self):
        clean_instance = CleanData(self.df)
        self.df = clean_instance.data_getter()

    def calculate_churn_rate(self):
        churn_rate = (self.df['Churn'].sum()/len(self.df)*100).round(0)
        print(f'Churn Rate: {churn_rate} %')

    def gender_analysis(self):
        self.df['Churn_label'] = self.df['Churn'].map({1: 'Yes', 0: 'No'})
        self.df['gender_label'] = self.df['gender'].map({1: 'Male', 0: 'Female'})
        crosstab(self.df['gender_label'], self.df['Churn_label']).plot(kind='bar')
        plt.title('Gender vs Churn')
        plt.ylabel('Count')
        plt.show()

    def internet_service_analysis(self):
        total_customers = self.df['InternetService'].value_counts()
        churned_customers = self.df[self.df['Churn'] == 1]['InternetService'].value_counts()

        order = ['DSL', 'Fiber optic', 'No']
        total_customers = total_customers.reindex(order)
        churned_customers = churned_customers.reindex(order)

        fig, ax = plt.subplots(figsize=(8, 8))

        # Outer ring — Total customers
        ax.pie(
            total_customers,
            radius=1,
            autopct='%1.1f%%',
            pctdistance=0.85,
            wedgeprops=dict(width=0.3, edgecolor='w')
        )

        # Inner ring — Churned customers
        ax.pie(
            churned_customers,
            radius=0.7,
            autopct='%1.1f%%',
            pctdistance=0.75,
            wedgeprops=dict(width=0.3, edgecolor='w')
        )

        # Center text
        ax.text(0, 0, 'Total vs Churn', ha='center', va='center', fontsize=12, weight='bold')

        # Legend
        ax.legend(
            order,
            title='Internet Service',
            loc='center left',
            bbox_to_anchor=(1, 0.5)
        )

        plt.title('Internet Service Distribution: Total vs Churn')
        plt.tight_layout()
        plt.show()

    def numeric_features_analysis(self):
        summary = (
            self.df.groupby('Churn')[['tenure', 'MonthlyCharges', 'TotalCharges']]
            .agg(['mean'])
            .round(2)
        )

        # صاف کردن ستون‌ها
        summary.columns = ['_'.join(col) for col in summary.columns]

        # رسم جدول
        fig, ax = plt.subplots(figsize=(10, 3))
        ax.axis('off')

        table = ax.table(
            cellText=summary.values,
            colLabels=summary.columns,
            rowLabels=summary.index,
            cellLoc='center',
            loc='center'
        )

        # تنظیم فونت
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 1.6)

        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4C72B0')
            elif col == -1:
                cell.set_text_props(weight='bold')
                cell.set_facecolor('#EAEAF2')
            else:
                cell.set_facecolor('#F7F7F7')

        plt.title(
            "Churn Relationship with Tenure, Monthly Charges, and Total Charges",
            fontsize=13,
            weight='bold',
            pad=12
        )

        plt.show()

        numerical_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

        for col in numerical_cols:
            plt.figure(figsize=(6, 4))
            sns.histplot(data=self.df, x=col, hue='Churn', bins=30, kde=True)
            plt.title(f'Distribution of {col} by Churn')
            plt.show()

    def export_data(self):
        self.df = self.df.drop(['Churn_label', 'gender_label'], axis=1)
        self.df.to_excel(self.DATA_PATH + 'churn_data_clean.xlsx', index=False)

    def explor_process(self):
        self.read_data()
        self.clean_data()
        self.calculate_churn_rate()
        self.gender_analysis()
        self.internet_service_analysis()
        self.numeric_features_analysis()
        self.export_data()

e = ExploratoryAnalysis()
e.explor_process()
