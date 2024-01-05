
# Predicting Bank Loan Approvals using Machine Learning

Banks provide loans and credit products to both individual consumers and businesses in order to generate interest income. However, there is significant financial risk if applicants default or fail to pay back on time. Manual underwriting done via extensive paperwork and human evaluation often has low accuracy and high costs. This project aims to develop an automated loan approval prediction system using machine learning for enhanced speed, precision and standardization while lowering costs.

## Dataset 

The dataset for this project is sourced from this [Kaggle repository](https://www.kaggle.com/datasets/vikramamin/bank-loan-approval-lr-dt-rf-and-auc). It contains over 10,000 observations on past loan applicants with details across demographic, financial history, credit report and requested loan attributes. The target variable indicates if the applied loan was approved or the application rejected.

## Data Import & Storage

The CSV dataset is imported into a Pandas dataframe which allows efficient manipulation using vectorized operations. For large datasets, a SQL database would be preferred for scalable storage and querying. Distribution analysis during exploration may reveal the need for big data warehousing technologies like PySpark if the data volume grows significantly. 

## Exploratory Data Analysis

Gaining insight into datasets through graphical and statistical analysis is crucial before applying machine learning algorithms which can pick up unexpected patterns or biases. Key aspects analyzed are:

**Univariate Distribution:** 

- Histogram and density plots for numeric variables
- Bar plots for categorical features
- Descriptive stats for central tendency, dispersion, shape

**Bivariate Analysis**

- Scatterplots between features  
- Correlation matrix heatmap 
- Groupwise aggregation and segmented analysis

**Multivariate Relationships** 

- Dimensionality reduction via PCA 
- Clustering using K-Means to find patterns

**Target Correlation**

- Compare group metrics between positive and negative classes through tables/graphs
- Statistical significance testing for difference in group means 

**Data Quality Assessment**

- Missing value identification
- Duplicate observation detection  
- Outlier detection using standard deviation thresholds
- Digital bias checking for uneven error rates among groups

**Feature Encoding** 

- Information lossiness evaluation for numeric encoding of categoricals

## Data Preparation

The pipeline for preparing the raw data for modeling consists of:

**Missing Value Imputation:** Missing values in features are filled using central tendency measures like mean/median/mode based on distribution type.

**Outlier Handling:** Extreme values skewing distributions are capped at chosen percentiles.

**Categorical Encoding:** Methods like one-hot encoding convert text categories into numeric without ordering assumptions.

**Feature Scaling:** All features are normalized to a standard range like 0-1 using min-max or z-score scaling for comparable magnitude.

**Feature Selection:** Redundant or irrelevant features are removed using statistical tests like mutual information or embedded methods like Lasso.

**Class Re-balancing:** As loan default is a rare event, the model may benefit from re-sampling minority class for balanced training. 

**Train-Test Split:** Final dataset is divided into mutually exclusive training and holdout test sets for unbiased evaluation.

## Model Development 

The following predictive models are trained and optimized for loan approval classification:

**Logistic Regression:** Linear model suitable for numerically-driven decisions and probabilistic predictions. Regularization handles high dimensionality of one-hot encoded data.

**Random Forest:** Ensemble model of decision trees providing high accuracy, capture of feature interactions and avoids overfitting. Number of trees, tree depth and leaf splits are tuned.  

**XGBoost:** State-of-the-art gradient boosted decision tree algorithm with advances like column subsampling for tabular data. Various regularization hyperparameters provide precision control. 

A stratified K-Fold cross validation methodology is employed for tuning model hyperparameters in a stable, unbiased manner by evaluating on different data slices. Grid, random search with warm restarts or Bayesian methods explore the tuning space efficiently to find optimal combinations of hyperparameters maximizing predictive performance.

## Model Evaluation

**Performance Metrics:** Along with ROC AUC, precision-recall curve and PR AUC provide a more detailed view into algorithm behavior on the imbalanced dataset. Additional metrics like information score can help assess real-world utility. 

**Learning Curves:** Validation performance vs training set size plots highlight whether more data would significantly help. Plateaus indicate diminishing returns.

**Confusion Matrices:** Break down helps visualize precise tradeoffs between different types of correct and incorrect predictions. Insights can guide techniques improving specific classes.

**Individual Condition Analysis:** SIMPLE model explanations from TreeInterpreter help identify precise demographic, financial and loan conditions influential in predictions at an individual applicant level for debugging bad decisions.

**Bias Checking:** Group and pairwise metrics between models reveal uneven accuracy affecting certain minority demographics disproportionately. Mitigation requires regularization during modeling.

The evaluation provides a 360-degree view into model behavior beyond aggregate performance, highlighting issues and further improvement opportunities.

## Author

Jigyansu Rout