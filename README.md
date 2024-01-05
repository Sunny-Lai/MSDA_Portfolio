# MSDA Portfolio
This is a collection of my favorite projects I have completed during my coursework towards a Masters of Science in Data Analytics at Western Governors University.
## Table of Contents
- [Forecasting Hospital Readmissions: A KNN Classification Analysis](#forecasting-hospital-readmissions-a-knn-classification-analysis)
- [Prescription Medication Interrelations: A Market Basket Analysis](#prescription-medication-interrelations-a-market-basket-analysis)
- [Unraveling Sentiment Patterns: A Recurrent Neural Network Analysis](#unraveling-sentiment-patterns-a-recurrent-neural-network-analysis)
## [Forecasting Hospital Readmissions: A KNN Classification Analysis](https://github.com/Sunny-Lai/MSDA_Portfolio/tree/main/KNN_Classification)
#### Objective
The goal of this analysis is to utilize k-nearest neighbors (KNN) to identify what factors most significantly relate to patients within a medical database being readmitted within 30 days. The variables that will be analyzed are within a dataset provided by the University.
#### Technology Used
- Python (Pandas, Seaborm, Matplotlib, Numpy)
    - SkLearn (KNeighborsClassifier, GridSearch, StandardScaler)
- Jupyter Notebook
#### Data Cleaning and Preparation
- Importing Packages, Labraries and Dataset
- Detecting for duplicates, missing values, and outliers
- Data Wrangling (transforming categorical values to numerical with dummy variables)
- Exploratory Data Analysis
#### Analysis
- Scaling Data
- Feature Selection
- Splitting and training the data
- Determining K
- Classification model, Confusion Matrix, and ROC Curve
#### Results
The accuracy scores that resulted provide confidence in our ROC KNN classification model. With a precision score of 98 percent, the model shows it can accurately predict if a patient is readmitted within 30 days of discharge 98 percent of the time with the predictor variables provided. The accuracy scores that resulted from indicates this model as a strong classifier of data that can produce accurate resuts based on our input variables with confidence.
## [Prescription Medication Interrelations: A Market Basket Analysis](https://github.com/Sunny-Lai/MSDA_Portfolio/tree/main/Market_Basket_Analysis)
#### Objective
The goal of this analysis is to utilize market basket analysis to determine what medications are most correlated with the medication abilfy. A Medical Database of patient prescriptions provided by the university was utilized for this Analysis.  Creating a model that can determine medications that are commonly purchased together provides stakeholders with the ability to predict what medications patients may need and conduct further research for the cause of these correlations. 
#### Technology Used
- Python (Numpy, Pandas, Matplotlib, Seaborn)
    - MlExtend (Transaction Encoder, Apriori, Association Rules)
- Juypter Notebook
#### Data Cleaning and Preparation
- Cleaning process: Detection and removal of missing values and duplicates.
- Transformation method: Employed a transaction encoder package to convert the dataset into a logical data frame.
#### Analysis
- Applied association rules package for filtering and pruning based on specific parameters.
- Utilized the Apriori algorithm to calculate item purchase frequencies.
- Parameters for filtering:
    - Lift: Indicates likelihood of consequent when antecedent is present.
    - Support: Measures frequency of itemset occurrence in dataset.
    - Confidence: Measures likelihood of one itemset occurring if another is present.
- Identified top three rules from association table based on these criteria.
#### Results
To answer our research question of determining the most correlated medications with Abilify, we need to further filter our analysis. To accomplish this, I filtered for the medication Abilify as either the antecedent or consequent and sorted by lift. I chose lift as the main sorting measurement as values of lift greater than one have an increased correlation. Evaluating the medication with the highest lift value in relationship to Abilify, we can see that the medication is metformin. The lift of this itemset is 1.91, indicating a high correlation that a person will purchase metformin in addition to abilify. The support of this itemset is 0.023, indicating the itemset frequency within this dataset is 2.3 percent. The confidence of this itemset is the proportion of prescriptions that include the itemset divided by the proportion of just the antecedent. Analyzing abilify as the antecedent and the consequent, the support is higher at 0.46, or 46 percent.
## [Unraveling Sentiment Patterns: A Recurrent Neural Network Analysis](https://github.com/Sunny-Lai/MSDA_Portfolio/tree/main/Recurrent_Neural_Network)
#### Objective
The goal of this analysis is to utilize recurrent neural networking and natural language processing on a combined dataset of reviews from Amazon, IMDB, and Yelp to detect patterns and make predictions that provide actionable insight for stakeholders.
#### Technology Used
- Python (Numpy, Matplotlib, Pandas, Seaborn)
    - TensorFlow, NLTK, SkLearn, Keras
- Jupyter Notebook
#### Data Cleaning and Preparation
- Read and import the datasets into the Jupyter notebook and combine the datasets into one data frame 
- Conduct data cleaning and exploratory analysis by checking for shape, missing values, vocabulary size, unusual characters, embedding length 
- Clean text by lowercasing, removing punctuation, and removing unwanted characters 
- Implement stopwords to remove conjunctions and word particles 
- Lemmatize and tokenize the data  
- Vectorize the data with integer encoding 
- Add padding to either before or after the sequences 
- Split the dataset into validation, training, and test sets into an 80/20 split, as this is the common partitioning utilized for machine learning
#### Analysis
To create our neural network model, I imported the library TensorFlow and utilized the package keras. Keras is a package in Python that is utilized to build, configure, and deploy neural networks.
- Model Parameters: Our neural network model consists of 3 layers, which are the embedding, LSTM, and dense layers. The embedding layer is used within natural language processing to create embeddings, or numerical vector representations, for categorical variables, this layer consists of 48,600 trainable parameters. The Long Short-Term Memory (LSTM) layer assists our neural network model to process long-term dependencies in sequential data, and this layer consists of 16,800 trainable parameters. The dense layer allows our neural network model to process complex patterns and embeddings through linear and non-linear transformations, and this layer consists of 61 trainable parameters.
#### Results
- Utilizing our stopping criteria, our model was run with a various number of epochs until our stopping criteria halts. This resulted when we ran 20 epochs, the stopping criteria halted our model at 12 epochs when our validation loss did not improve.
- Our model was processed through a various number of stopping criteria and epochs. The best predictive accuracy we achieved was 47.09 %. This indicates we were able to correctly predict 47.09% of training outcomes. Our average loss was 0.6937.
- The architecture of our RNN model is well suited for text classification and structured for the consideration of our binary sentiment values. Though the prediction accuracy was approximately 50%, we were still able to predict whether a user's review was positive or negative based on test classification. This will provide actionable insights for stakeholders to improve customer satisfaction.
