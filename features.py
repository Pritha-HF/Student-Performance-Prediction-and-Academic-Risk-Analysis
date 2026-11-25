pip install flask scikit-learn pandas numpy matplotlib seaborn joblib

from sklearn.base import BaseEstimator, TransformerMixin

class AcademicFeatureEngineer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self

    def transform(self, X):
        d = X.copy()
        d['Continuous_Avg'] = (d['Assignment_Marks'] + d['Internal_Marks']) / 2.0
        d['Engagement_Index'] = (d['Attendance'] * d['Study_Hours']) / 100.0
        d['Academic_Momentum'] = d['Continuous_Avg'] - d['Previous_Marks']
        return d
