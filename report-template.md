# Module 12 Report Template

## Overview of the Analysis

In this section, describe the analysis you completed for the machine learning models used in this Challenge. This might include:

* Explain the purpose of the analysis.
* Explain what financial information the data was on, and what you needed to predict.
* Provide basic information about the variables you were trying to predict (e.g., `value_counts`).
* Describe the stages of the machine learning process you went through as part of this analysis.
* Briefly touch on any methods you used (e.g., `LogisticRegression`, or any resampling method).

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

### Stages of the machine learning process as part of the analysis


### The Logistic Regression method
Train_test_split

The data is split into different sets to train and to test. 
 
***(import logistic regression model)***


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Logistic Regression Model with Original Data):
  * The accuracy score 
  * The precision of the model 
  * Recall is 



* Machine Learning Model 2 (Logistic Regression Model with Resampled Data):
  * The accuracy score 
  * The precision of the model 
  * Recall is 

## Summary

Summarize the results of the machine learning models, and include a recommendation on the model to use, if any. For example:
* Which one seems to perform best? How do you know it performs best?
* Does performance depend on the problem we are trying to solve? (For example, is it more important to predict the `1`'s, or predict the `0`'s? )

If you do not recommend any of the models, please justify your reasoning.

