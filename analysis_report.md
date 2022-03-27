## Overview of the Analysis
---

### Purpose of the analysis

The purpose of this analysis is to use various techniques to train and evaluate models with imbalanced classes. The `lending_data.csv` dataset provides historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. This analysis consist of three subsections: 

* Split the Data into Training and Testing sets
* Create a Logistic Regression Model with the Original Data
* Predict a Logistic Regression Model with Resampled Training Data


### Financial Information Data

Financial information data from the `lending_data.csv` file includes data on loan size, interest rate, borrower income, total debt, loan status, etc. 

One main aspect to the data is the loan status. Values in this item are `0` or `1`. The value of `0` means that the loan is healthy. A value of `1` means that the loan has a high risk of defaulting. 


### Information about variables

Data are split into training and testing sets in our model:
* label set (y) from the "loan_status" column  
* features (X) DataFrame from the remaining columns

Using the `value_counts` function to check for class imbalance of the value counts. 


### Stages of the machine learning process as part of the analysis
Stage of the machine learning process:
1. Create a model 
`model = LogisticRegression(randon_state=1)`

2. Fit the model using training data
`model.fit(X_train, y_train)`

3. Test the model
`predictions = model.predict(X_test)`

4. Evaluate the model performance
* `balanced_accuracy_score`- in binary and multiclass classification problems to deal with imbalanced datasets.
* `confusion_matrix` - evaluates how the model performed. Shows the number of obervations that the model correctly classified by telling us the number of true positives and true negative values. 
* `classification_report` Calculates the accuracy, precision, recall, and F1 scores for each class. 


### The Logistic Regression method
Train_test_split

The data is split into different sets to train and to test. 
 
Using the `LogisticRegression` classifier:
***(import logistic regression model)***


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Logistic Regression Model with Original Data):
  * The accuracy score 95.2% (0.952).
  * The precision of the model 85% (0.85).
  * Recall is 91% (0.91).


* Machine Learning Model 2 (Logistic Regression Model with Resampled Data):
  * The accuracy score 99.3% (0.993).
  * The precision of the model 84% (0.84).
  * Recall is 99% (0.99).

## Summary

The precision score was slightly higher with Model 1 than Model 2, with 85% and 84%, respectively. This score represents the percent of prediction were made correctly. The recall was higher in Model 2 with 99% versus Model 1 91%. The recall show the percent of the positive cases were caught in the model. It appears that Model 2, logistic regression model with resampled training data, has a higher accuracy score to predict with accounts are health loans and which are high-risk loans with 99.3% versus 95.2% with Model 1. 
