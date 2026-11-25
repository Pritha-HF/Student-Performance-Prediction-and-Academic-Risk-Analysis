import json
import warnings
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, NearestNeighbors
from sklearn.cluster import KMeans
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    silhouette_score, classification_report
)
from features import AcademicFeatureEngineer

def generate_student_dataset(n_samples=1500, seed=42):
    rng = np.random.default_rng(seed)
    attendance = np.clip(rng.normal(78, 14, n_samples), 35, 100)
    study_hours = np.clip(rng.normal(14, 6, n_samples), 1, 35)
    previous_marks = np.clip(rng.normal(66, 15, n_samples), 20, 99)
    assignment_marks = np.clip(0.55 * previous_marks + 0.35 * attendance + rng.normal(0, 8, n_samples), 15, 100)
    internal_marks = np.clip(0.50 * previous_marks + 0.30 * assignment_marks + 0.45 * study_hours + rng.normal(0, 7, n_samples), 15, 100)
    learning_support = rng.choice(['Low', 'Medium', 'High'], size=n_samples, p=[0.25, 0.50, 0.25])
    support_bonus = np.where(learning_support == 'High', 4.5, np.where(learning_support == 'Medium', 1.5, -2.5))

    final_marks = (
        0.22 * attendance
        + 0.75 * study_hours
        + 0.26 * previous_marks
        + 0.22 * assignment_marks
        + 0.28 * internal_marks
        + support_bonus
        - 12.0
        + rng.normal(0, 4.5, n_samples)
    )
    final_marks = np.clip(final_marks, 10, 100)

    df = pd.DataFrame({
        'Student_ID': [f'STU{1001 + i}' for i in range(n_samples)],
        'Attendance': np.round(attendance, 1),
        'Study_Hours': np.round(study_hours, 1),
        'Previous_Marks': np.round(previous_marks, 1),
        'Assignment_Marks': np.round(assignment_marks, 1),
        'Internal_Marks': np.round(internal_marks, 1),
        'Learning_Support': learning_support,
        'Final_Exam_Marks': np.round(final_marks, 1)
    })

    # Inject realistic data quality issues for preprocessing demonstration
    missing_idx = rng.choice(n_samples, size=25, replace=False)
    df.loc[missing_idx[:12], 'Attendance'] = np.nan
    df.loc[missing_idx[12:], 'Study_Hours'] = np.nan
    df.loc[rng.choice(n_samples, size=5, replace=False), 'Study_Hours'] = 95.0
    df = pd.concat([df, df.iloc[:10]], ignore_index=True)
    df.to_csv('student_performance.csv', index=False)
    return df

print("=== 1. COLLECTING DATA ===")
df = generate_student_dataset()
print("Raw Dataset Shape:", df.shape)
print("Missing Values Before Cleaning:\n", df.isnull().sum())
print("Duplicate Rows Before Cleaning:", df.duplicated().sum())

print("\n=== 2. PREPROCESSING DATA ===")
df = df.drop_duplicates().reset_index(drop=True)

for col in ['Attendance', 'Study_Hours', 'Previous_Marks', 'Assignment_Marks', 'Internal_Marks']:
    df[col] = df[col].fillna(df[col].median())
    q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df[col] = df[col].clip(lower=max(0, lower), upper=upper)

df['Attendance'] = df['Attendance'].clip(0, 100)
df['Pass_Fail'] = (df['Final_Exam_Marks'] >= 45.0).astype(int)

print("Cleaned Dataset Shape:", df.shape)
print("Pass/Fail Distribution:\n", df['Pass_Fail'].value_counts())

print("\n=== 3. ANALYZING DATA (EDA) ===")
num_eda_cols = ['Attendance', 'Study_Hours', 'Previous_Marks', 'Assignment_Marks', 'Internal_Marks', 'Final_Exam_Marks']
print(df[num_eda_cols].describe().T.round(2))

plt.figure(figsize=(8, 6))
sns.heatmap(df[num_eda_cols].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix: Academic Factors vs Final Exam Marks')
plt.tight_layout()
plt.savefig('eda_correlation_heatmap.png', dpi=150)
plt.close()

feature_cols = ['Attendance', 'Study_Hours', 'Previous_Marks', 'Assignment_Marks', 'Internal_Marks', 'Learning_Support']
X = df[feature_cols]
y_reg = df['Final_Exam_Marks']
y_clf = df['Pass_Fail']

X_train, X_test, y_reg_train, y_reg_test, y_clf_train, y_clf_test = train_test_split(
    X, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
)

numerical_features = [
    'Attendance', 'Study_Hours', 'Previous_Marks', 'Assignment_Marks', 'Internal_Marks',
    'Continuous_Avg', 'Engagement_Index', 'Academic_Momentum'
]
categorical_features = ['Learning_Support']

def make_preprocessor():
    return ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', Pipeline([
                ('imputer', SimpleImputer(strategy='most_frequent')),
                ('ohe', OneHotEncoder(drop='first', handle_unknown='ignore'))
            ]), categorical_features)
        ]
    )

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

print("\n=== 4A. LINEAR REGRESSION (PREDICTING FINAL MARKS) ===")
base_linreg = Pipeline([
    ('fe', AcademicFeatureEngineer()),
    ('prep', make_preprocessor()),
    ('model', LinearRegression())
])
base_cv_r2 = cross_val_score(base_linreg, X_train, y_reg_train, cv=kf, scoring='r2').mean()

ridge_pipe = Pipeline([
    ('fe', AcademicFeatureEngineer()),
    ('prep', make_preprocessor()),
    ('model', Ridge())
])
ridge_grid = GridSearchCV(ridge_pipe, {'model__alpha': [0.01, 0.1, 1.0, 5.0, 10.0]}, cv=kf, scoring='r2', n_jobs=-1)
ridge_grid.fit(X_train, y_reg_train)
best_reg_model = ridge_grid.best_estimator_

y_reg_pred = np.clip(best_reg_model.predict(X_test), 0, 100)
reg_r2 = r2_score(y_reg_test, y_reg_pred)
reg_mae = mean_absolute_error(y_reg_test, y_reg_pred)
reg_rmse = np.sqrt(mean_squared_error(y_reg_test, y_reg_pred))
print(f"Baseline Linear Regression CV R2: {base_cv_r2:.4f}")
print(f"Optimized Ridge Regression Best Params: {ridge_grid.best_params_} | Test R2: {reg_r2:.4f} | MAE: {reg_mae:.2f} | RMSE: {reg_rmse:.2f}")

print("\n=== 4B. LOGISTIC REGRESSION (PASS/FAIL CLASSIFICATION) ===")
log_pipe = Pipeline([
    ('fe', AcademicFeatureEngineer()),
    ('prep', make_preprocessor()),
    ('model', LogisticRegression(max_iter=1000, random_state=42))
])
log_grid = GridSearchCV(
    log_pipe,
    {'model__C': [0.05, 0.1, 1.0, 5.0, 10.0], 'model__class_weight': [None, 'balanced']},
    cv=skf, scoring='f1', n_jobs=-1
)
log_grid.fit(X_train, y_clf_train)
best_log_model = log_grid.best_estimator_

y_log_pred = best_log_model.predict(X_test)
log_acc = accuracy_score(y_clf_test, y_log_pred)
log_f1 = f1_score(y_clf_test, y_log_pred)
print(f"Optimized Logistic Regression Best Params: {log_grid.best_params_}")
print(f"Test Accuracy: {log_acc:.4f} | Test F1-Score: {log_f1:.4f}")
print(classification_report(y_clf_test, y_log_pred, target_names=['Fail (0)', 'Pass (1)']))

print("\n=== 4C. KNN (CLASSIFICATION & SIMILAR STUDENT PATTERNS) ===")
knn_pipe = Pipeline([
    ('fe', AcademicFeatureEngineer()),
    ('prep', make_preprocessor()),
    ('model', KNeighborsClassifier())
])
knn_grid = GridSearchCV(
    knn_pipe,
    {'model__n_neighbors': [5, 9, 13, 17, 21], 'model__weights': ['uniform', 'distance']},
    cv=skf, scoring='f1', n_jobs=-1
)
knn_grid.fit(X_train, y_clf_train)
best_knn_model = knn_grid.best_estimator_

y_knn_pred = best_knn_model.predict(X_test)
knn_acc = accuracy_score(y_clf_test, y_knn_pred)
knn_f1 = f1_score(y_clf_test, y_knn_pred)
print(f"Optimized KNN Best Params: {knn_grid.best_params_}")
print(f"KNN Test Accuracy: {knn_acc:.4f} | Test F1-Score: {knn_f1:.4f}")

# Fit NearestNeighbors on training feature space to find similar peer profiles
nn_preprocessor = Pipeline([
    ('fe', AcademicFeatureEngineer()),
    ('prep', make_preprocessor())
])
X_train_scaled = nn_preprocessor.fit_transform(X_train)
nn_finder = NearestNeighbors(n_neighbors=3, metric='euclidean')
nn_finder.fit(X_train_scaled)

reference_students = X_train.copy()
reference_students['Final_Exam_Marks'] = y_reg_train.values
reference_students['Status'] = np.where(y_clf_train.values == 1, 'Pass', 'Fail')
reference_students = reference_students.reset_index(drop=True)

print("\n=== 4D. K-MEANS CLUSTERING (STUDENT SEGMENTATION) ===")
kmeans_pipe = Pipeline([
    ('fe', AcademicFeatureEngineer()),
    ('prep', make_preprocessor()),
    ('kmeans', KMeans(n_clusters=3, random_state=42, n_init=20))
])
clusters_train = kmeans_pipe.fit_predict(X_train)
X_cluster_scaled = kmeans_pipe[:-1].transform(X_train)
sil_score = silhouette_score(X_cluster_scaled, clusters_train)

cluster_means = pd.Series(y_reg_train.values).groupby(clusters_train).mean().sort_values()
cluster_label_map = {
    int(cluster_means.index[0]): 'Academically At-Risk',
    int(cluster_means.index[1]): 'Average',
    int(cluster_means.index[2]): 'High Performing'
}
print(f"K-Means Silhouette Score: {sil_score:.4f}")
print("Cluster Mapping (by mean exam score):", {cluster_label_map[k]: round(v, 2) for k, v in cluster_means.items()})

print("\n=== 5. MAJOR FACTORS AFFECTING PERFORMANCE ===")
raw_num_cols = ['Attendance', 'Study_Hours', 'Previous_Marks', 'Assignment_Marks', 'Internal_Marks']
factor_pipe = Pipeline([('scaler', StandardScaler()), ('lin', LinearRegression())])
factor_pipe.fit(X_train[raw_num_cols], y_reg_train)
coefs = np.abs(factor_pipe.named_steps['lin'].coef_)
coef_pct = (coefs / coefs.sum()) * 100
major_factors = sorted(
    [{'feature': col.replace('_', ' '), 'impact_pct': round(float(pct), 1)} for col, pct in zip(raw_num_cols, coef_pct)],
    key=lambda x: x['impact_pct'],
    reverse=True
)
for f in major_factors:
    print(f"  {f['feature']}: {f['impact_pct']}% relative impact")

bundle = {
    'reg_model': best_reg_model,
    'log_model': best_log_model,
    'knn_model': best_knn_model,
    'nn_preprocessor': nn_preprocessor,
    'nn_finder': nn_finder,
    'reference_students': reference_students,
    'kmeans_pipe': kmeans_pipe,
    'cluster_label_map': cluster_label_map
}
joblib.dump(bundle, 'student_models.joblib')

artifacts = {
    'metrics': {
        'reg_r2': round(reg_r2 * 100, 2),
        'reg_mae': round(reg_mae, 2),
        'log_acc': round(log_acc * 100, 2),
        'log_f1': round(log_f1 * 100, 2),
        'knn_acc': round(knn_acc * 100, 2),
        'silhouette': round(sil_score, 3)
    },
    'major_factors': major_factors
}
with open('model_artifacts.json', 'w') as f:
    json.dump(artifacts, f, indent=2)

print("\nSaved 'student_models.joblib', 'model_artifacts.json', 'student_performance.csv', and 'eda_correlation_heatmap.png'.")
