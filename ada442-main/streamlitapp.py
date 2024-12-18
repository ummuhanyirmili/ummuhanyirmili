import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import chi2, f_classif
from sklearn.metrics import (classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay)
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# Set Streamlit Page Configurations
st.set_page_config(page_title="Bank Marketing Analysis", layout="wide")
st.title("Bank Marketing Dataset Analysis")

# Sidebar Navigation
sections = [
    "About Data",
    "1. Data Cleaning",
    "2. Data Preprocessing",
    "3. Feature Selection",
    "4. Model Selection",
    "5. Hyperparameter Tuning",
    "6. Model Evaluation"
]
selected_section = st.sidebar.radio("Choose a section:", sections)

# Load Data
data = pd.read_csv("bank-additional.csv", delimiter=';')
numeric_data = data.select_dtypes(include=[np.number])
corr_matrix = numeric_data.corr()

if selected_section == "About Data":
    st.header("About Data")
    st.write("This dataset contains information about a bank's direct marketing campaigns. It includes various attributes about the customer and the outcome of the campaign.")
    st.write("Here is a summary of the data:")
    st.dataframe(data.head())
    st.text(data.info())
    st.write(data.describe())

if selected_section == "1. Data Cleaning":
    st.header("1. Data Cleaning")

    column_order = ["age", "job", "marital", "education", "default", "housing", "loan",
                    "contact", "month", "day_of_week", "duration", "campaign", "pdays",
                    "previous", "poutcome", "emp.var.rate", "cons.price.idx",
                    "cons.conf.idx", "euribor3m", "nr.employed", "y"]

    bank_additional = pd.read_csv("bank-additional.csv", delimiter=';', names=column_order)
    st.write("Shape of the dataset:", bank_additional.shape)
    st.write("Here is the first few rows of the dataset:")
    st.dataframe(bank_additional.head())

    st.write("Checking for missing values:")
    st.write(data.isnull().sum())

if selected_section == "2. Data Preprocessing":
    st.header("2. Data Preprocessing")

    # Convert categorical columns to 'category' dtype
    categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
    data[categorical_columns] = data[categorical_columns].astype('category')
    st.write("Converted categorical columns to 'category' dtype.")

    # Scale numerical columns
    scaler = MinMaxScaler()
    numerical_cols = ['age', 'duration', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']
    data[numerical_cols] = scaler.fit_transform(data[numerical_cols])
    st.write("Applied Min-Max Scaling to numerical columns.")
    st.dataframe(data.head())

    # Create 'contacted_before' binary column
    data['contacted_before'] = data['pdays'].apply(lambda x: 0 if x == 999 else 1)
    data.drop('pdays', axis=1, inplace=True)
    st.write("Created 'contacted_before' binary feature from 'pdays'.")
    st.dataframe(data.head())

    # One-hot encode categorical columns
    data = pd.get_dummies(data, columns=categorical_columns)
    st.write("Applied one-hot encoding to categorical columns.")
    st.dataframe(data.head())

if selected_section == "3. Feature Selection":
    st.header("3. Feature Selection")

    # Label encode 'y' column
    le = LabelEncoder()
    data['y'] = le.fit_transform(data['y'])
    st.write("Encoded target column 'y'.")

    # Convert all columns to numeric where possible
    data = data.apply(pd.to_numeric, errors='coerce')  # Convert non-numeric columns to NaN

    # Select only numeric columns for correlation calculation
    numeric_data = data.select_dtypes(include=[np.number])

    # Compute the correlation matrix
    corr_matrix = numeric_data.corr()
    threshold = 0.1
    important_features = corr_matrix.index[abs(corr_matrix['y']) > threshold].tolist()
    if 'y' in important_features:
        important_features.remove('y')

    st.write("Features selected based on correlation with target variable:", important_features)



if selected_section == "4. Model Selection":
    st.header("4. Model Selection")
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    numeric_data = data.select_dtypes(include=[np.number])
    # Compute the correlation matrix
    corr_matrix = numeric_data.corr()
    threshold = 0.1
    st.write(corr_matrix)
    # Ensure that important_features is defined before using it
    important_features = corr_matrix.index[abs(corr_matrix['y']) > threshold].tolist()
    st.write(important_features)
    X = data[important_features]
    y = data['y']
    

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

    # Define model pipelines here
    from imblearn.pipeline import Pipeline as ImbPipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.neural_network import MLPClassifier
    from imblearn.over_sampling import SMOTE

    pipelines = {
        'Logistic Regression': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', MinMaxScaler()),
            ('classifier', LogisticRegression())
        ]),
        'Random Forest': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('classifier', RandomForestClassifier())
        ]),
        'Neural Network': ImbPipeline([
            ('smote', SMOTE(random_state=42)),
            ('scaler', MinMaxScaler()),
            ('classifier', MLPClassifier(max_iter=1000))
        ])
    }

    # Train and evaluate models
    results = {}
    for name, pipeline in pipelines.items():
        pipeline.fit(X_train, y_train)
        predictions = pipeline.predict(X_test)
        results[name] = classification_report(y_test, predictions, output_dict=True)

    for name, metrics in results.items():
        st.write(f"{name} Classification Report:")
        st.json(metrics)

if selected_section == "5. Hyperparameter Tuning":
    st.header("5. Hyperparameter Tuning")

    param_grid_lr = {
        'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'classifier__penalty': ['l2']
    }

    param_grid_rf = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_features': ['sqrt', 'log2'],
        'classifier__max_depth': [None, 10, 20, 30],
        'classifier__min_samples_split': [2, 5, 10]
    }

    st.write("Hyperparameter grids defined for Logistic Regression and Random Forest.")
    st.json({"Logistic Regression": param_grid_lr, "Random Forest": param_grid_rf})

if selected_section == "6. Model Evaluation":
    #st.write(data)
    data['y'] = data['y'].map({'yes': 1, 'no': 0})
    numeric_data = data.select_dtypes(include=[np.number])
    # Compute the correlation matrix
    corr_matrix = numeric_data.corr()
    threshold = 0.1
    important_features = corr_matrix.index[abs(corr_matrix['y']) > threshold].tolist() 
    if 'y' in important_features:
        important_features.remove('y')
    X = data[important_features]
    y = data['y']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    st.header("6. Model Evaluation")

    best_lr_model = LogisticRegression(C=1, penalty='l2')
    best_rf_model = RandomForestClassifier(n_estimators=100, max_depth=None)

    best_lr_model.fit(X_train, y_train)
    best_rf_model.fit(X_train, y_train)

    lr_predictions = best_lr_model.predict(X_test)
    rf_predictions = best_rf_model.predict(X_test)

    lr_cm = confusion_matrix(y_test, lr_predictions)
    rf_cm = confusion_matrix(y_test, rf_predictions)

    st.write("Logistic Regression Confusion Matrix:")
    st.write(lr_cm)

    st.write("Random Forest Confusion Matrix:")
    st.write(rf_cm)

    st.write("Classification Reports:")
    st.text("Logistic Regression")
    st.text(classification_report(y_test, lr_predictions))

    st.text("Random Forest")
    st.text(classification_report(y_test, rf_predictions))

    # Display Confusion Matrices
    st.write("Logistic Regression Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(lr_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    plt.tight_layout()
    st.pyplot(fig)

    st.write("Random Forest Confusion Matrix:")
    fig, ax = plt.subplots()
    sns.heatmap(rf_cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    st.pyplot(fig)
    st.markdown('<p style="font-size:100px">&#128640;</p>', unsafe_allow_html=True)

