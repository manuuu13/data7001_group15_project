import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

# 1. Load data
df = pd.read_csv('mhp_dataset.csv')

# 2. Data Cleaning & Encoding
# Target: CGPA (Categorical to Numeric)
le = LabelEncoder()
df['cgpa_label'] = le.fit_transform(df['cgpa'].astype(str))

# Features selection
features = ['stress_z_score', 'anxiety_z_score', 'depression_z_score', 'age', 'gender']
X = df[features].copy()
y = df['cgpa_label']

# Encode categorical features for the model
X['age'] = LabelEncoder().fit_transform(X['age'].astype(str))
X['gender'] = LabelEncoder().fit_transform(X['gender'].astype(str))

# 3. Dataset Splitting (Train: 70%, Validation: 15%, Test: 15%)
# Step 1: Split out the test set
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
# Step 2: Split the remaining into Train and Validation
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1765, random_state=42, stratify=y_train_full
)

print(f"Data Split: Train({len(X_train)}), Val({len(X_val)}), Test({len(X_test)})")

# 4. Automated Hyperparameter Tuning with Cross-Validation
# Using GridSearchCV to find the best configuration
rf = RandomForestClassifier(random_state=42)
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10, None],
    'min_samples_split': [2, 5]
}

# 5-Fold Cross Validation on the training set
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf = grid_search.best_estimator_
print(f"Best Parameters: {grid_search.best_params_}")

# 5. Model Evaluation
# Validate on the Validation Set
val_preds = best_rf.predict(X_val)
print(f"Validation Accuracy: {accuracy_score(y_val, val_preds):.4f}")

# Final Assessment on the unseen Test Set
test_preds = best_rf.predict(X_test)
print("\nFinal Test Set Performance:")
print(classification_report(y_test, test_preds, target_names=le.classes_))