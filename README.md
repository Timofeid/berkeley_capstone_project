# Credit Card Approval Prediction
Capstone project for UC Berkeley AI/ML certification program

## Open questions
- Review Feature development approach
- Null values for Occupation column is about 30%. Drop column, drop rows, try to fill in. Current implementation: filled in based on NAME_INCOME_TYPE average
- Review model performance metrics: Performance for models is very similar to baseline (dummy model where everyone is "Good" client). Are there ways for improvement?

## Executive summary
TBD

### Research Question
How can we accurately predict whether a credit card application will be approved based on an applicant’s financial and personal information?

### Rationale
This question is crucial because it addresses the efficiency and accuracy of credit card approval processes. By developing a predictive model, financial institutions can make more informed decisions, reduce the risk of approving unsuitable candidates, and improve customer satisfaction by speeding up the approval process. The insights gained from this project will be translated into actionable business intelligence, providing clear, data-driven recommendations that non-AI/ML personnel can implement to enhance their credit card approval strategies.

This project aims to bridge the gap between complex machine learning models and practical business applications, ensuring that the results are both understandable and actionable for stakeholders.

### Data Sources
[Credit Card Approval Prediction from Kaggle](https://www.kaggle.com/datasets/rikdifos/credit-card-approval-prediction)

## Methodology

### Exploratory Data Analysis
Exploratory data analysis focuses on understanding the dataset’s structure, identifying missing values, detecting outliers, and discovering patterns and correlations among features. This includes visualizations and statistical summaries to ensure the data is clean and ready for modeling.
1. Understand the Dataset

	* Load the Data: Import the dataset and inspect the first few rows to get an overview.
	* Data Types: Check the data types of each column to understand what kind of data you are dealing with (e.g., numerical, categorical).

2. Data Cleaning

	* Missing Values: Identify missing values and decide how to handle them (e.g., imputation, removal).
	* Outliers: Detect outliers that might skew the analysis and decide on appropriate handling methods.
	* Duplicates: Check for and remove any duplicate records.

3. Data Analysis
    * Correlation Matrix: Compute and visualize the correlation matrix to see relationships between numerical features.
	* Scatter Plots: Create scatter plots for pairs of numerical features to visually inspect relationships.
	* Cross-tabulation: For categorical features, create cross-tabulations to understand their interactions.

4. Feature Engineering

	* Feature Creation: Based on initial findings, create new features that might enhance model performance.
	* Transformation: Apply necessary transformations (e.g., log transformation) to normalize skewed data.

TBD: Describe data challenges and how it was solved.

### Model selection

#### Model selection approach
1.	Define relevant performance metrics such as accuracy, precision, recall, F1 score, and ROC-AUC to evaluate the models.
2.	Establish a baseline metric, which provides a point of reference for evaluating the performance of more complex models.
3.	Split the dataset into training and testing sets to assess the models’ performance on unseen data.
4.	Train and evaluate multiple machine learning models, including:
	* K-Nearest Neighbors (KNN)
	* Logistic Regression
	* Support Vector Machine (SVM)
	* Decision Trees
	* Random Forest
	* Deep Neural Networks
5.	Compare the models’ performance using the specified metrics and visualize the results for clear comparison.
6.	Select the best-performing model based on the comparative analysis of performance metrics.
7.	Interpret the chosen model using appropriate techniques such as feature importance to ensure transparency and understanding of the model’s decision-making process.

#### Relevant performance metric
From a business perspective, the most important metric for a credit card approval model should be the F1 score. This metric balances precision and recall, making it particularly valuable for credit card approval decisions where both false positives (approving an unqualified applicant) and false negatives (rejecting a qualified applicant) carry significant business implications.

##### Why the F1 Score is Critical:

1.	Precision (Positive Predictive Value): High precision means that when the model predicts an approval, it is likely to be correct. This reduces the risk of approving applicants who may default or not meet the financial criteria, thereby mitigating financial loss.
2.	Recall (Sensitivity or True Positive Rate): High recall ensures that most qualified applicants are approved, maximizing customer acquisition and satisfaction. Missing out on qualified applicants can lead to lost business opportunities.
3.	F1 Score: By balancing precision and recall, the F1 score provides a comprehensive measure of the model’s performance. It is especially useful in scenarios where the cost of false positives and false negatives are both high, as it ensures that the model maintains a good trade-off between avoiding unqualified applicants and not missing out on qualified ones.

##### Additional Considerations:

* ROC-AUC: This metric can also be important as it measures the model’s ability to discriminate between positive and negative classes across different threshold settings. A high ROC-AUC indicates a good overall performance of the model.
* Business Rules and Risk Management: Besides F1 score, integrating business-specific rules and risk management strategies into the model’s deployment phase can further refine decision-making, ensuring that the model aligns with the company’s financial risk tolerance and regulatory requirements.

By focusing on the F1 score, businesses can ensure that their credit card approval model effectively identifies qualified applicants while minimizing the risk associated with approving unqualified ones. This balanced approach is crucial for maintaining financial health and customer satisfaction.

#### Results
What did your research find?

#### Next steps
What suggestions do you have for next steps?

#### Outline of project

- [Link to notebook 1]()
- [Link to notebook 2]()
- [Link to notebook 3]()


##### Contact and Further Information
