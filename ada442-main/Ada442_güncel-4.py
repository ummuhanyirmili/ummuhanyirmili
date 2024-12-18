#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, chi2, f_classif

from sklearn.metrics import classification_report, accuracy_score, confusion_matrix,ConfusionMatrixDisplay, precision_recall_fscore_support, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE


# # ABOUT DATA

# In[2]:


data = pd.read_csv("bank-additional.csv", delimiter=';')

data.info()
data.describe()


# # 1. Data Cleaning

# In[3]:


column_order = ["age", "job", "marital", "education", "default", "housing", "loan",
                "contact", "month", "day_of_week", "duration", "campaign", "pdays",
                "previous", "poutcome", "emp.var.rate", "cons.price.idx",
                "cons.conf.idx", "euribor3m", "nr.employed", "y"]

# Read the CSV file with the specified column order
bank_additional = pd.read_csv("bank-additional.csv", delimiter=';', names=column_order)


# In[30]:


print(bank_additional.shape)


# In[4]:


print(bank_additional.head())


# In[5]:


print(data.isnull().sum())


# In[6]:


categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'day_of_week', 'poutcome']
data[categorical_columns] = data[categorical_columns].astype('category')


# In[7]:


reference_column = 'age'
numerical_columns = ['duration', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed']
fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.scatterplot(data=data, x=reference_column, y=column, ax=axes[i], color = (16/255, 37/255, 81/255))
    axes[i].set_title(f'{reference_column} vs {column}', fontsize=12, color = (16/255, 37/255, 81/255))
    axes[i].set_xlabel(reference_column, fontsize=10)
    axes[i].set_ylabel(column, fontsize=10)

for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[8]:


numerical_columns = ['duration', 'campaign', 'pdays', 'previous',
                     'emp.var.rate', 'cons.price.idx', 'cons.conf.idx',
                     'euribor3m', 'nr.employed']

fig, axes = plt.subplots(3, 4, figsize=(18, 12))
axes = axes.flatten()

for i, column in enumerate(numerical_columns):
    sns.boxplot(
        data=data,
        y=column,
        ax=axes[i],
        boxprops=dict(facecolor = (16/255, 37/255, 81/255), edgecolor = (16/255, 37/255, 81/255)),
        medianprops=dict(color="black", linewidth=2)
    )
    axes[i].set_title(f'Distribution of {column}', fontsize=12,color = (16/255, 37/255, 81/255) )
    axes[i].set_ylabel(column, fontsize=10)
    axes[i].grid(axis='y', linestyle='--', alpha=0.7)

for j in range(len(numerical_columns), len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.show()


# In[9]:


# Cap the values in the 'campaign' and 'duration' columns at the 95th percentile
for column in ['campaign', 'duration']:
    threshold = data[column].quantile(0.90)
    data[column] = np.where(data[column] > threshold, threshold, data[column])

# Remove any rows where the 'age' column has values greater than 90
data = data[data['age'] <= 90]

# Print the first few rows to check the transformed data
data.head(5)


# # 2. Data Preprocessing

# In[11]:


scaler = MinMaxScaler()

# List of numerical columns to scale (update as needed)
numerical_cols = ['age', 'duration', 'campaign', 'emp.var.rate', 'cons.price.idx', 'cons.conf.idx', 'euribor3m', 'nr.employed']

# Apply Min-Max Scaling
data[numerical_cols] = scaler.fit_transform(data[numerical_cols])

# Print the first few rows to check the transformed data
data.head()


# In[12]:


# Convert the 'pdays' column into a binary feature: 0 indicates not contacted (999), and 1 indicates contacted
data['contacted_before'] = data['pdays'].apply(lambda x: 0 if x == 999 else 1)

# Remove the original 'pdays' column from the DataFrame
data.drop('pdays', axis=1, inplace=True)

# Display the first few rows of the new 'contacted_before' column to confirm the changes
print(data[['contacted_before']].head())


# In[13]:


# Define the list of categorical columns for one-hot encoding
categorical_columns = ['job', 'marital', 'education', 'default', 'housing',
                       'loan', 'contact', 'month', 'day_of_week', 'poutcome']

# Identify the columns that are present in the DataFrame
existing_columns = [col for col in categorical_columns if col in data.columns]

# Display missing columns for troubleshooting
missing_columns = [col for col in categorical_columns if col not in data.columns]
if missing_columns:
    print(f"The following columns are not found in the DataFrame and will be skipped: {missing_columns}")

# Perform one-hot encoding on the available columns
if existing_columns:
    data = pd.get_dummies(data, columns=existing_columns)
    print("One-hot encoding has been successfully completed.")
else:
    print("No valid categorical columns found for one-hot encoding.")

# Display the updated DataFrame structure
print(data.head())


# # 3. Feature Selection

# In[15]:


# Check if 'y' exists in the DataFrame and its data type
print("Column 'y' exists:", 'y' in data.columns)
print("Data type of 'y':", data['y'].dtype if 'y' in data.columns else "Column 'y' does not exist.")

# If 'y' exists and needs to be encoded:
if 'y' in data.columns and data['y'].dtype == 'object':
    from sklearn.preprocessing import LabelEncoder
    le = LabelEncoder()
    data['y'] = le.fit_transform(data['y'])
    print("Column 'y' encoded successfully.")


# In[16]:


# Ensure all categorical variables are one-hot encoded
data = pd.get_dummies(data)

# Calculate the correlation matrix
corr_matrix = data.corr()

# Define a threshold for selecting the features based on correlation
threshold = 0.1  # Adjust as needed

# Find and store features that have a correlation above the threshold with 'y'
important_features = corr_matrix.index[abs(corr_matrix['y']) > threshold].tolist()

# Remove 'y' from the list if it's included
if 'y' in important_features:
    important_features.remove('y')

# Print the important features
print("Important features based on correlation:", important_features)

print("Important features based on correlation: 'duration', 'previous', 'emp.var.rate', 'euribor3m', 'nr.employed', 'contacted_before', 'contact_cellular', 'contact_telephone', 'month_dec', 'month_mar', 'month_may', 'month_oct', 'month_sep', 'poutcome_nonexistent', 'poutcome_success' ")
# # 4. Model Selection

# In[17]:


# Assuming data[important_features] and data['y'] are already defined
X = data[important_features]
y = data['y']

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Define pipelines for each model with SMOTE integration
pipelines = {
    'logistic_regression': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', MinMaxScaler()),
        ('classifier', LogisticRegression())
    ]),
    'random_forest': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('classifier', RandomForestClassifier())
    ]),
    'neural_network': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('scaler', MinMaxScaler()),
        ('classifier', MLPClassifier(max_iter=1000))
    ])
}

# Fit and evaluate each model
results = {}
for name, pipeline in pipelines.items():
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    results[name] = classification_report(y_test, predictions)

# Print results
for name, result in results.items():
    print(f"{name} Classification Report:\n{result}\n")


# # 5. Hyperparameter Tuning

# In[18]:


# Hyperparameter grid for Logistic Regression
param_grid_lr = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'classifier__penalty': ['l2']  # Norm used in penalization
}

# Hyperparameter grid for Random Forest
param_grid_rf = {
    'classifier__n_estimators': [50, 100, 200],  # Number of trees in the forest
    'classifier__max_features': ['sqrt', 'log2'],  # Number of features to consider at every split
    'classifier__max_depth': [None, 10, 20, 30],  # Maximum number of levels in tree
    'classifier__min_samples_split': [2, 5, 10]  # Minimum number of samples required to split a node
}


# In[19]:


# Logistic Regression Grid Search
grid_search_lr = GridSearchCV(
    estimator=pipelines['logistic_regression'],  # your logistic regression pipeline
    param_grid=param_grid_lr,
    scoring='accuracy',  # or other relevant scoring method
    cv=5,  # number of cross-validation folds
    verbose=1,  # for logging output
    n_jobs=-1  # number of CPU cores to use
)

# Random Forest Grid Search
grid_search_rf = GridSearchCV(
    estimator=pipelines['random_forest'],  # your random forest pipeline
    param_grid=param_grid_rf,
    scoring='accuracy',  # or other relevant scoring method
    cv=5,  # number of cross-validation folds
    verbose=1,  # for logging output
    n_jobs=-1  # number of CPU cores to use
)


# In[20]:


# Fit Grid Search for Logistic Regression
grid_search_lr.fit(X_train, y_train)

# Fit Grid Search for Random Forest
grid_search_rf.fit(X_train, y_train)


# In[21]:


print("Best parameters for Logistic Regression:", grid_search_lr.best_params_)
print("Best score for Logistic Regression:", grid_search_lr.best_score_)

print("Best parameters for Random Forest:", grid_search_rf.best_params_)
print("Best score for Random Forest:", grid_search_rf.best_score_)


# # 6. Model Evaluation

# In[22]:


# Evaluate the models with the best parameters
best_lr_model = grid_search_lr.best_estimator_
best_rf_model = grid_search_rf.best_estimator_

lr_predictions = best_lr_model.predict(X_test)
cm = confusion_matrix(y_test, lr_predictions)
rf_predictions = best_rf_model.predict(X_test)
cm_2 = confusion_matrix(y_test, rf_predictions)


# Accuracy
lr_accuracy = accuracy_score(y_test, lr_predictions)
rf_accuracy = accuracy_score(y_test, rf_predictions)

# Classification Report
print("Logistic Regression Classification Report:")
print(classification_report(y_test, lr_predictions))

print("Random Forest Classification Report:")
print(classification_report(y_test, rf_predictions))

# Confusion Matrix
print("Logistic Regression Confusion Matrix:")
print(confusion_matrix(y_test, lr_predictions))

print("Random Forest Confusion Matrix:")
print(confusion_matrix(y_test, rf_predictions))


# In[23]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.title('Confusion Matrix for Logistic Regression')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[24]:


plt.figure(figsize=(8, 6))
sns.heatmap(cm_2, annot=True, fmt="d", cmap='Blues', xticklabels=['Predicted No', 'Predicted Yes'], yticklabels=['Actual No', 'Actual Yes'])
plt.title('Confusion Matrix for Random Forest')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()


# In[ ]:





# In[ ]:




