#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Fraud Detection with Logistic Regression & XGBoost

# Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier


# In[2]:


# Step 2: Load Dataset
df = pd.read_csv("creditcard.csv")
print("Original shape:", df.shape)


# In[3]:


# Step 3: Preprocess
scaler = StandardScaler()
df['Amount'] = scaler.fit_transform(df[['Amount']])
df.drop('Time', axis=1, inplace=True)


# In[4]:


# Step 4: Split Features/Labels
X = df.drop('Class', axis=1)
y = df['Class']


# In[5]:


# Step 5: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)


# In[6]:


# Step 6: Apply SMOTE to balance the training set
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

print("Class distribution after SMOTE:\n", y_res.value_counts())


# In[8]:


# Step 7: Logistic Regression
lr = LogisticRegression(max_iter=1000)
lr.fit(X_res, y_res)
y_pred_lr = lr.predict(X_test)
y_scores_lr = lr.predict_proba(X_test)[:, 1]

print("\n🔎 Logistic Regression Results:")
print(confusion_matrix(y_test, y_pred_lr))
print(classification_report(y_test, y_pred_lr))
print("ROC AUC Score:", roc_auc_score(y_test, y_scores_lr))


# In[9]:


# Step 8: XGBoost Classifier
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
xgb.fit(X_res, y_res)
y_pred_xgb = xgb.predict(X_test)
y_scores_xgb = xgb.predict_proba(X_test)[:, 1]

print("\n🔎 XGBoost Results:")
print(confusion_matrix(y_test, y_pred_xgb))
print(classification_report(y_test, y_pred_xgb))
print("ROC AUC Score:", roc_auc_score(y_test, y_scores_xgb))


# In[10]:


# Step 9: Compare ROC Curves
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_scores_xgb)

plt.figure(figsize=(8,6))
plt.plot(fpr_lr, tpr_lr, label="Logistic Regression")
plt.plot(fpr_xgb, tpr_xgb, label="XGBoost")
plt.plot([0,1], [0,1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.grid()
plt.show()


# In[ ]:




