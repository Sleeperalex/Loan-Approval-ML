# Home Loan Approval Prediction Project

## Overview
The **Home Loan Approval Prediction Project** aims to enhance the loan approval process in the financial sector by leveraging machine learning. The goal is to predict whether a loan application should be approved (1) or rejected (0) based on applicant attributes such as income, loan amount, credit history, and more. By building a robust predictive model, this project seeks to reduce risks for lenders, promote fairer loan assessments, and improve decision-making processes.
https://www.kaggle.com/datasets/rishikeshkonapure/home-loan-approval/data

---

## Project Stages

### Stage 1: Data Preprocessing and Exploratory Data Analysis (EDA)
- **Objective**: Clean and preprocess the dataset to handle missing values, encode categorical variables, and standardize numerical features.
- **Key Steps**:
  - Handle missing values by imputing with mean/mode or using advanced techniques.
  - Remove outliers to improve model stability and data quality.
  - Perform EDA to visualize key patterns, identify class imbalance, and detect potential outliers.
- **Limitations**: 
  - Initial use of Logistic Regression revealed poor recall for class 0 (rejected loans) due to class imbalance and non-linearity in the data.

### Stage 2: Model Evaluation and Ensemble Methods
- **Objective**: Evaluate multiple machine learning models for binary classification, including Logistic Regression, Decision Tree, Random Forest, and Naive Bayes.
- **Key Steps**:
  - Train and evaluate models using metrics like precision, recall, and F1-score.
  - Explore ensemble methods (e.g., hard voting) to aggregate the strengths of individual models.
- **Limitations**:
  - Moderate recall for class 1 (approved loans) and imbalance in precision and recall for the minority class (class 0).
  - Ensemble methods improved performance but still relied heavily on majority voting, which can dilute the contributions of more accurate models.

### Stage 3: Advanced Models and Improved Ensemble
- **Objective**: Incorporate advanced models like CatBoost and LightGBM to handle class imbalance and improve recall for class 0.
- **Key Steps**:
  - Train and evaluate advanced models alongside traditional classifiers.
  - Combine these models in an ensemble to improve overall performance.
- **Results**:
  - Improved recall for class 0, demonstrating the value of exploring beyond familiar methods.

---

## Final Stage: Boosted Ensemble Approach

### Problem Formalization
- **Goal**: Predict whether a loan should be approved (1) or rejected (0) based on applicant details and financial factors.
- **Algorithm**: 
  - A boosted ensemble approach using **XGBoost**, **CatBoost**, and **LightGBM**, combined with traditional models (Logistic Regression, Decision Tree, Random Forest, Naive Bayes).
- **Key Steps**:
  1. **Data Preprocessing**: Handle missing values, remove outliers, and address class imbalance using SMOTE and class weights.
  2. **Model Training**: Train models with carefully tuned hyperparameters using grid search.
  3. **Evaluation**: Use cross-validation, ROC curves, and metrics like recall to evaluate performance comprehensively.

### Limitations
- **Model Complexity**: Boosted models, while accurate, can be computationally intensive and less interpretable.
- **Overfitting Risk**: Improper tuning can lead to overfitting, especially on small datasets.
- **Class Imbalance**: SMOTE and boosting help address class imbalance but may introduce noise or biases.
- **Scalability**: Training time may increase with additional preprocessing steps and parameter tuning.

---

## Methodology

### 1. Data Preprocessing
- **Handling Missing Values**: Missing values were imputed using mean/mode for numerical and categorical columns, respectively.
- **Outlier Removal**: Outliers were removed using the Interquartile Range (IQR) method.
- **Class Imbalance**: Addressed using **SMOTE** (Synthetic Minority Oversampling Technique) and class weights.
- **Encoding**: Categorical variables were encoded using **LabelEncoder**.

### 2. Data Visualization
- **Correlation Matrix**: Visualized relationships between features to identify key predictors.
- **Pair Plot**: Analyzed relationships between numerical variables, colored by loan status.
- **Distribution Plots**: Visualized the distribution of numerical features to understand data characteristics.

### 3. Model Training and Evaluation
- **Models Used**: Logistic Regression, Decision Tree, Random Forest, Naive Bayes, XGBoost, LightGBM, CatBoost.
- **Grid Search**: Hyperparameter tuning was performed using **GridSearchCV** to optimize for recall for class 0.
- **Evaluation Metrics**: Precision, recall, F1-score, ROC-AUC, and confusion matrices were used to evaluate model performance.

### 4. Ensemble Learning
- **Voting Classifier**: A hard voting ensemble was created using the best-performing models (LightGBM, CatBoost, XGBoost, Random Forest, Decision Tree, Naive Bayes).
- **Results**: The ensemble achieved a recall of 78% for class 0 (rejected loans), with an overall accuracy of 77%.

---

## Results

### Classification Reports
- **Best Models**: Random Forest, CatBoost, and XGBoost consistently performed well, with high recall for class 0.
- **Ensemble Model**: Achieved a recall of 78% for class 0, reducing the risk of approving ineligible applicants.

### ROC Curves
- **AUC Scores**: Random Forest achieved the highest AUC score (0.83), followed closely by CatBoost and XGBoost (0.81 each).
- **ROC Analysis**: Tree-based models outperformed Logistic Regression and Naive Bayes, demonstrating better classification performance.

### Confusion Matrices
- **False Positives**: The ensemble model misclassified 14 ineligible applicants as approved, which remains a concern.
- **False Negatives**: 35 approved loans were misclassified as rejected, which is less critical compared to false positives.

---

## Discussion and Conclusion

### Business Problem
- **Critical Goal**: Accurately identify rejected loan applications (class 0) to minimize financial risks.
- **Recall for Class 0**: The ensemble model achieved a recall of 78%, effectively reducing false negatives for rejected loans.
- **False Positives**: 14 ineligible applicants were misclassified as approved, which poses a financial risk to the business.

### Strengths
- The ensemble model offers a solid recall for class 0, reducing the risk of approving ineligible applicants.
- Models like Random Forest, CatBoost, and XGBoost consistently perform well in recall for class 0 across validation folds.

### Weaknesses
- The 14 false positives (class 0 misclassified as class 1) indicate that some ineligible applicants could still be approved.
- Logistic Regression and Naive Bayes perform poorly overall and fail to provide adequate recall for class 0.

### Final Recommendation
- **Ensemble Model**: The ensemble model, with its high recall for class 0 (78%), is the most suitable for this problem. It aligns with the business priority of minimizing financial risks associated with incorrectly approved loans.
