# MSDA_Portfolio
A Portfolio of my projects towards completion in a Masters of Science in Data Analytics 


## KNN Classification Project
### Objective
The goal of this analysis is to utilize k-nearest neighbors (KNN) to identify what factors most significantly relate to patients within a medical database being readmitted within 30 days. The variables that will be analyzed are within a dataset provided by the University.
### Technology Used
- Python
- Jupyter Notebook
### Data Cleaning and Preparation
- Importing Packages, Labraries and Dataset
- Detecting for duplicates, missing values, and outliers
- Data Wrangling (transforming categorical values to numerical with dummy variables)
- Exploratory Data Analysis
### Analysis
- Scaling Data
- Feature Selection
- Splitting and training the data
- Determining K
- Classification model, Confusion Matrix, and ROC Curve
### Results
The accuracy scores that resulted provide confidence in our ROC KNN classification model. With a precision score of 98 percent, the model shows it can accurately predict if a patient is readmitted within 30 days of discharge 98 percent of the time with the predictor variables provided. The accuracy scores that resulted from indicates this model as a strong classifier of data that can produce accurate resuts based on our input variables with confidence.
## Market Basket Analysis Project
### Objective
The goal of this analysis is to utilize market basket analysis to determine what medications are most correlated with the medication abilfy. A Medical Database of patient prescriptions provided by the university was utilized for this Analysis.  Creating a model that can determine medications that are commonly purchased together provides stakeholders with the ability to predict what medications patients may need and conduct further research for the cause of these correlations. 
### Technology Used
- Python (Numpy, Pandas, Matplotlib, Seaborn)
    - MlExtend (Transaction Encoder, Apriori, Association Rules)
- Juypter Notebook
### Data Cleaning and Preparation
- Cleaning process: Detection and removal of missing values and duplicates.
- Transformation method: Employed a transaction encoder package to convert the dataset into a logical data frame.
### Analysis
- Applied association rules package for filtering and pruning based on specific parameters.
- Utilized the Apriori algorithm to calculate item purchase frequencies.
- Parameters for filtering:
    - Lift: Indicates likelihood of consequent when antecedent is present.
    - Support: Measures frequency of itemset occurrence in dataset.
    - Confidence: Measures likelihood of one itemset occurring if another is present.
- Identified top three rules from association table based on these criteria.
### Results
To answer our research question of determining the most correlated medications with Abilify, we need to further filter our analysis. To accomplish this, I filtered for the medication Abilify as either the antecedent or consequent and sorted by lift. I chose lift as the main sorting measurement as values of lift greater than one have an increased correlation. Evaluating the medication with the highest lift value in relationship to Abilify, we can see that the medication is metformin. The lift of this itemset is 1.91, indicating a high correlation that a person will purchase metformin in addition to abilify. The support of this itemset is 0.023, indicating the itemset frequency within this dataset is 2.3 percent. The confidence of this itemset is the proportion of prescriptions that include the itemset divided by the proportion of just the antecedent. Analyzing abilify as the antecedent and the consequent, the support is higher at 0.46, or 46 percent.
