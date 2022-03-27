# credit_risk_resampling

Credit risk poses a classification problem that’s inherently imbalanced. This is because healthy loans easily outnumber risky loans. This application uses techniques to train and evaluate models with imbalanced classes. Using a dataset of historical lending activity from a peer-to-peer lending services company to build a model that can identify the creditworthiness of borrowers. 

Using the knowledge of the imbalanced-learn library, a logistic regression model is created to compare two versions of the dataset. First with the original dataset. Second, by resampling the data and using the RandomOverSampler module from the imbalanced-learn library.

In both cases, we will count of the target classes, train a logistic regression classifier, calculate the balanced accuracy score, generate a confusion matrix, and generate a classification report.

## Technologies
Click on the links below for documentation on each of the technologies used. This project uses the following libraries and dependencies:
+ [**Anaconda**](https://docs.anaconda.com/): an open source package and environment management system.
+ [**JupyterLab**](https://jupyterlab.readthedocs.io/en/stable/): an extensive environment using web-based user interface designed for data analysis. 
+ [**Pandas**](https://pandas.pydata.org/docs/getting_started/index.html): (included in Anaconda) a Python package data analysis toolkit.
+ [**Numpy**](https://numpy.org/doc/stable/) for scientific computing such as mathematical, basic statistical operations, and much more. 
+ [**scikitlearn**](https://scikit-learn.org/stable/install.html) an open source machine learning library that supports supervised and unsupervised learning. It also provides various tools for model fitting, data preprocessing, model selection, model evaluation, and many other utilities. 
+ [**imbalanced-learn**](https://imbalanced-learn.org/stable/): an open source, MIT-licensed libraby relying on skikit-learn and provides tools when dealing with classification with imbalanced classes. 
+ [**PyDotPlus**](https://pydotplus.readthedocs.io/): an improved version of the old pydot project that provides a Python Interface to Graphviz's Dot language. 

## Installation Guide
Check to ensure that Jupyterlab is installed on your machine by entering the following into your environment using `conda list`. If any of these techologies are not installed into your environment, use the corresponding codes in your terminal: 

```
pip install -U scikit-learn
conda install -c conda-forge imbalanced-learn
conda install -c conda-forge pydotplus
```

### Instructions to Use JupyterLab

To launch JupyterLab:
  1. Open terminal window, and type `conda activate dev`.
  2. Type `jupyter lab`. JupyterLab user interface should open in your browser. 
      a. Or copy and paste one of the URLs: "http://localhost:8888/lab?token=..." into web browser. 

To exit JupyterLab:
  1. On your web browser, use the Run button to shut down any running kernel sessions.
  2. On the menu bar, on the File menu, select Shut Down. 
  3. Okay to close tabs once a dialog box with "Server stopped" message indicates the server has stopped. 
  4. Navigate to terminal window, where you launched JupyterLab and type `conda deactivate`. This will return you to your `base` environment. 

## Usage 

1. Clone the repository `https://github.com/leighbadua/credit_risk_resampling.git`
2. Using your terminal, activate the environment and launch jupyter lab (instructions mentioned above). 
3. Launch `credit_risk_resampling.ipynb` in JupyterLab

### Import other required libraries and dependencies: 
<img width="443" alt="image" src="https://user-images.githubusercontent.com/96001018/160280551-4d8ec550-370d-47d7-bea0-3c318662910c.png">



## Overview of the Analysis


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
`Train_test_split` splits the data into different sets to train and to test. 

<img width="436" alt="image" src="https://user-images.githubusercontent.com/96001018/160280071-931b1f2c-fa52-4289-bd23-12be7e3588f5.png">

 
Using the `LogisticRegression` classifier:

<img width="388" alt="image" src="https://user-images.githubusercontent.com/96001018/160280092-4c5ed816-b481-458b-b704-c48afa31b225.png">


## Results

Using bulleted lists, describe the balanced accuracy scores and the precision and recall scores of all machine learning models.

* Machine Learning Model 1 (Logistic Regression Model with Original Data):
  * The accuracy score 95.2% (0.952).
  * The precision of the model 85% (0.85).
  * Recall is 91% (0.91).

<img width="470" alt="image" src="https://user-images.githubusercontent.com/96001018/160280146-efaa6bda-78db-4b40-baf0-2b8c5e3ebbe4.png">



* Machine Learning Model 2 (Logistic Regression Model with Resampled Data):
  * The accuracy score 99.3% (0.993).
  * The precision of the model 84% (0.84).
  * Recall is 99% (0.99).

<img width="490" alt="image" src="https://user-images.githubusercontent.com/96001018/160280160-4e17ae46-5f1b-4252-a81c-935e399a8317.png">



## Summary

The precision score was slightly higher with Model 1 than Model 2, with 85% and 84%, respectively. This score represents the percent of prediction were made correctly. The recall was higher in Model 2 with 99% versus Model 1 91%. The recall show the percent of the positive cases were caught in the model. It appears that Model 2, logistic regression model with resampled training data, has a higher accuracy score to predict with accounts are health loans and which are high-risk loans with 99.3% versus 95.2% with Model 1. 

## Contributors

Leigh Anne Badua leighbadua@gmail.com 


## License
MIT
