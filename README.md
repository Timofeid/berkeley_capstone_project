# Credit Card Approval Prediction
Capstone project for UC Berkeley AI/ML certification program

## Executive summary
#### Objective:
The research aimed to develop a robust predictive model for credit card approvals, leveraging applicants’ personal and financial data to streamline decision-making processes, mitigate risks, and enhance customer satisfaction.

#### Key Insights:

1.	**Data Enhancement:** We addressed significant data quality issues by imputing missing values in the OCCUPATION_TYPE column and removing outliers, irrelevant columns, and duplicates. This ensured the integrity and relevance of the data, critical for accurate model predictions.
2.	**Model Evaluation:** A variety of machine learning models were tested, including K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, Random Forests, XGBoost, Support Vector Machines (SVM), and Deep Neural Networks (DNN). The evaluation focused on the F1 score to balance the risk of false positives and false negatives, which is crucial for maintaining financial stability and customer trust.

#### Business Impact:
- Despite extensive testing, most models, including Logistic Regression, Random Forest, XGBoost, and DNN, did not surpass the performance of a baseline Dummy Classifier, indicating a predominant prediction of the majority class.
- The SVM model showed marginal improvement but did not provide significant discriminative capabilities, highlighting a gap in effectively differentiating between creditworthy and non-creditworthy applicants.

#### Recommendations:
To improve the predictive accuracy and utility of the credit card approval model, we recommend:

- Further Data Refinement: Enhance data quality through advanced feature engineering and handling of class imbalances.
- Advanced Model Tuning: Implement more sophisticated techniques such as hyperparameter optimization and ensemble methods to better capture the nuances of the data.
- Continuous Monitoring: Establish a robust evaluation framework to continuously assess model performance and adapt to changing market and applicant conditions.

#### Conclusion:
This research provides a foundational understanding and highlights areas for improvement in the credit card approval process. By addressing the identified challenges, financial institutions can develop more accurate and reliable models, ultimately improving decision-making efficiency and customer satisfaction.

## Research Question
How can we accurately predict whether a credit card application will be approved based on an applicant’s personal information?

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
	* Column removal: remove columns that do not provide significant information.

3. Data Analysis
    * Correlation Matrix: Compute and visualize the correlation matrix to see relationships between numerical features.
	* Scatter Plots: Create scatter plots for pairs of numerical features to visually inspect relationships.
	* Cross-tabulation: For categorical features, create cross-tabulations to understand their interactions.

4. Feature Engineering

	* Feature Creation: Based on initial findings, create new features that might enhance model performance.
	* Transformation: Apply necessary transformations (e.g., log transformation) to normalize skewed data.

#### EDA challenges

- Development of Target Label:
	- A target label was developed based on the individual’s financial history, which involved assessing and categorizing applicants according to predefined criteria. This label serves as the dependent variable for predictive modeling, distinguishing between applicants who meet specific financial criteria and those who do not.
- Handling Missing Values:
	- Approximately 30% of the values in the OCCUPATION_TYPE column were missing. To address this, missing values were imputed based on the average value within each Income_type category. This method ensures that the imputed values align with the financial characteristics typically associated with different income types, thereby maintaining the integrity and relevance of the data.


### Model selection

#### Model selection approach
1.	Define relevant performance metrics such as accuracy, precision, recall, F1 score, and ROC-AUC to evaluate the models.
2.	Establish a baseline metric, which provides a point of reference for evaluating the performance of more complex models.
3.	Split the dataset into training and testing sets to assess the models’ performance on unseen data.
4.	Train and evaluate multiple machine learning models, including:
	* K-Nearest Neighbors (KNN)
	* Logistic Regression
	* Support Vector Machine (SVM)
	* XGBoost
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

## Results
The credit card approval prediction project aimed to develop a robust machine learning model to enhance the efficiency and accuracy of the credit card application process. Throughout the project, we evaluated multiple models, including K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, Random Forests, XGBoost, Support Vector Machines (SVM), and a Deep Neural Network (DNN), alongside a baseline Dummy Classifier.

The following key findings were observed:

1.	***Model Performance:***
	- The majority of models, including the Logistic Regression, Random Forest, XGBoost, and DNN, exhibited performance metrics identical to the Dummy Classifier. This indicates that these models defaulted to predicting the majority class, failing to provide meaningful discrimination between approved and rejected applications.
	- The SVM showed marginally better performance with a slightly higher ROC AUC, yet the improvement was not significant, suggesting limited discriminative capability.
2.	***Evaluation Metrics:***
	- The models achieved high recall (1.000) across the board, which implies a strong tendency to predict the majority class. However, the ROC AUC scores of 0.500 indicate that the models, including the more complex DNN, did not effectively differentiate between the classes.
	- Precision and F1 scores mirrored the recall due to the models’ focus on the majority class, resulting in overall performance metrics that did not exceed the baseline set by the Dummy Classifier.
3.	***Challenges and Limitations:***
	- The uniformity in performance metrics across all models suggests potential issues such as imbalanced data, insufficient feature engineering, or inadequate model tuning. These challenges may have hindered the models’ ability to learn meaningful patterns from the data.

## Next steps
To address the limitations encountered and improve the model’s predictive capabilities, the following recommendations are proposed:

1.	**Data Enhancement:**
	- Further exploration into the data, including feature selection, creation, and engineering, is essential to provide the models with more informative attributes that can help in distinguishing between approved and rejected applications.
2.	**Model Tuning and Selection:**
	- Conduct more extensive hyperparameter tuning, especially for complex models like XGBoost and DNNs, to optimize their performance. Consider using more advanced techniques such as ensemble learning or stacking to leverage the strengths of multiple models.
3.	**Addressing Class Imbalance:**
	- Implement strategies to handle class imbalance, such as oversampling the minority class, undersampling the majority class, or using advanced techniques like SMOTE (Synthetic Minority Over-sampling Technique) to create a more balanced training set.
4.	**Evaluation and Monitoring:**
	- Continue monitoring the model’s performance with new data and refine the model as needed. Establish a robust evaluation framework that includes metrics beyond accuracy and recall to ensure comprehensive performance assessment.

In conclusion, while the current models did not surpass the baseline set by the Dummy Classifier, this project has laid the groundwork for future improvements. By addressing the highlighted challenges and implementing the recommended strategies, we can enhance the model’s ability to make accurate and reliable predictions, ultimately optimizing the credit card approval process for better business outcomes.
