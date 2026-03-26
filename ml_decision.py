import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, classification_report

def load_and_preprocess_data(filepath):
    df = pd.read_csv(filepath)
    
    # Handle missing values
    # Numerical features
    num_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if 'loan_status' in num_cols:
        num_cols = num_cols.drop('loan_status') # removing target variable
    imputer = SimpleImputer(strategy='median')
    df[num_cols] = imputer.fit_transform(df[num_cols])
    
    # Categorical features
    cat_cols = df.select_dtypes(include=['object']).columns
    cat_imputer = SimpleImputer(strategy='most_frequent')
    df[cat_cols] = cat_imputer.fit_transform(df[cat_cols])
    
    # Encode categorical variables
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        
    X = df.drop('loan_status', axis=1)
    y = df['loan_status']
    
    return X, y

def main():
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data('credit_risk_dataset.csv')
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("Training 3 Machine Learning Models...\n")
    
    # Model 1: Logistic Regression
    lr = LogisticRegression(random_state=42, max_iter=1000)
    lr.fit(X_train_scaled, y_train)
    lr_preds = lr.predict(X_test_scaled)
    print("--- 1. Logistic Regression ---")
    print(f"Accuracy: {accuracy_score(y_test, lr_preds):.4f}")
    
    # Model 2: Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train)
    rf_preds = rf.predict(X_test_scaled)
    print("\n--- 2. Random Forest ---")
    print(f"Accuracy: {accuracy_score(y_test, rf_preds):.4f}")
    
    # Model 3: Gradient Boosting
    gb = GradientBoostingClassifier(n_estimators=100, random_state=42)
    gb.fit(X_train_scaled, y_train)
    gb_preds = gb.predict(X_test_scaled)
    print("\n--- 3. Gradient Boosting ---")
    print(f"Accuracy: {accuracy_score(y_test, gb_preds):.4f}")
    
    print("\nTraining Ensemble (Voting) Classifier for making final decision...")
    # Making final decision using a Voting Classifier (Hard Voting)
    voting_clf = VotingClassifier(
        estimators=[('lr', lr), ('rf', rf), ('gb', gb)],
        voting='hard'
    )
    voting_clf.fit(X_train_scaled, y_train)
    voting_preds = voting_clf.predict(X_test_scaled)
    
    print("\n--- Final Decision (Ensemble Voting) ---")
    print(f"Accuracy: {accuracy_score(y_test, voting_preds):.4f}")
    print("\nClassification Report for the Decision Maker:\n")
    print(classification_report(y_test, voting_preds))
    
    print("\nThe Ensemble model combines the 3 ML models to make a more robust, final decision on loan status.")
    
    # Example prediction for a single sample
    print("\n--- Example Prediction ---")
    sample = X_test_scaled[0].reshape(1, -1)
    prediction = voting_clf.predict(sample)
    decision = "Approved (Low Risk)" if prediction[0] == 0 else "Rejected (High Risk)"
    print(f"Sample prediction using the Ensemble Model: {decision}")

if __name__ == "__main__":
    main()
