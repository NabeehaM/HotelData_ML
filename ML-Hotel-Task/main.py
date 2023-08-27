import pandas as pd
import seaborn as sns
import matplotlib as plt
from matplotlib import figure
from matplotlib import pyplot as pyplt
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge

import numpy as np
def l1Regularisation(df):
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    #print(np.unique(y))
    # Split the data into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    alpha = 1.0  # Regularization strength (adjust as needed)
    model = Lasso(alpha=alpha)

    model.fit(x_train, y_train)

    y_pred = model.predict(x_test)
    print(np.unique(y_pred))
    print(np.unique(y_test))
    a = accuracy_score(y_test, y_pred)
    print("L1 Reg Accuracy Score")
    print(a)

def knnModel(df):
    # Create a k-NN classifier with k=3
    knn = KNeighborsClassifier(n_neighbors=3)
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test.values)
    a = accuracy_score(y_test, y_pred)
    print("KNN Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("KNN F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("KNN Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("KNN Recall Score")
    print(recall)

def logisticRegression(df):
    print("Logistic Regression")
    model = LogisticRegression()
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("Logistic Regression Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("Logistic Regression F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("Logistic Regression Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("Logistic Regression Recall Score")
    print(recall)

def svmClassifier(df):
    # Create the SVM classifier
    num_rows_to_keep = 5000 #Subset the Data because it is taking too long to train (takes 20 minutes)
    random_sample_df = df.sample(n=num_rows_to_keep,random_state=42)
    df = random_sample_df.copy()

    svm_classifier = SVC(kernel='linear', C=1.0, decision_function_shape='ovr')
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Train the classifier on the training data
    svm_classifier.fit(x_train, y_train)
    # Make predictions on the test set
    y_pred = svm_classifier.predict(x_test)
    y_pred = svm_classifier.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("SVM Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("SVM F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("SVM Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("SVM Recall Score")
    print(recall)

def randomForestWithAdaBoost(df):
    print("Random Forest with Ada boost")
    # Create a decision tree as the base estimator
    base_tree = RandomForestClassifier(max_depth=1)
    # Create an AdaBoost classifier
    adaboost_classifier = AdaBoostClassifier(estimator=base_tree, n_estimators=50, random_state=42)
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    # Train the AdaBoost classifier
    adaboost_classifier.fit(x_train, y_train)
    # Make predictions
    y_pred = adaboost_classifier.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("RF HPT Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("RF HPT F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("RF HPT Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, pos_label='Yes')
    print("RF HPT Recall Score")
    print(recall)
def randomForestWithGridSearch(df):
    num_rows_to_keep = 5000  # Subset the Data because it is taking too long to train (takes 20 minutes)
    random_sample_df = df.sample(n=num_rows_to_keep, random_state=42)
    df = random_sample_df.copy()

    print("Random Forest with Grid Search")
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': [1.0, 'sqrt']
    }
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    grid_search = GridSearchCV(rf_classifier, param_grid, cv=5)
    grid_search.fit(x_train, y_train)
    y_pred = grid_search.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("RF GridSearch Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("RF GridSearch F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("RF GridSearch Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("RF GridSearch Recall Score")
    print(recall)

    # Print the best hyperparameters
    print("Best Hyperparameters:", grid_search.best_params_)

def randomForestWithRandomSearch(df):
    print("Random Search")
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    # Instantiate RandomizedSearchCV

    num_rows_to_keep = 2000  # Subset the Data because it is taking too long to train (takes 20 minutes)
    random_sample_df = df.sample(n=num_rows_to_keep, random_state=42)
    df = random_sample_df.copy()

    print("Random Forest with Random Search")
    param_grid = {
        'max_depth': [None, 10, 20],
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': [1.0, 'sqrt']
    }

    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    random_search = RandomizedSearchCV(rf_classifier, param_distributions=param_grid, n_iter=10, cv=5,
                                       )
    random_search.fit(x_train, y_train)
    y_pred = random_search.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("RF RandomSearch Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("RF RandomSearch F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("RF RandomSearch Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("RF RandomSearch Recall Score")
    print(recall)

    # Print the best hyperparameters
    print("Best Hyperparameters:", random_search.best_params_)


def  randomForest(df):
    # Create the Random Forest classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']
    # Train the model
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    rf_classifier.fit(x_train, y_train)
    y_pred = rf_classifier.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("RF Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred, pos_label='Yes')
    print("RF F1 Score")
    print(f)

    precision = precision_score(y_test, y_pred, pos_label='Yes')
    print("RF Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("RF Recall Score")
    print(recall)
def decisionTree(df):

    x = df.drop('is_canceled', axis=1)
    y = df['is_canceled']

    dt_classifier = DecisionTreeClassifier()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=47)
    dt_classifier.fit(x_train, y_train)
    fnames = x.columns.tolist()
    targetnames = y.unique().tolist()
    plt.figure.Figure(figsize=(15, 10))
    #fnames = x.columns.tolist()
    #targetnames = y.unique().tolist()
    plot_tree(dt_classifier,
              feature_names=fnames,
              class_names=targetnames, filled=True)
    pyplt.show()

    y_pred = dt_classifier.predict(x_test)
    a = accuracy_score(y_test, y_pred)
    print("Accuracy Score")
    print(a)

    f = f1_score(y_test, y_pred,pos_label='Yes')
    print("F1 Score")
    print(f)


    precision = precision_score(y_test, y_pred,pos_label='Yes')
    print("Precision Score")
    print(precision)

    recall = recall_score(y_test, y_pred, average='micro')
    print("Recall Score")
    print(recall)
    confusionmatrix = confusion_matrix(y_test, y_pred, labels=targetnames)

    # Create a heatmap visualization of the confusion matrix
    sns.heatmap(confusionmatrix, annot=True, fmt='g', cmap='Blues', xticklabels=targetnames, yticklabels=targetnames)

    # Add x and y labels to the confusion matrix
    pyplt.xlabel('Predicted label')
    pyplt.ylabel('True label')
    pyplt.show()

    # Get feature importances
    feature_importances = dt_classifier.feature_importances_

    # Plot feature importances
    plt.figure.Figure(figsize=(8, 6))
    pyplt.bar(x.columns, feature_importances)
    pyplt.xticks(rotation=45)
    pyplt.xlabel('Feature')
    pyplt.ylabel('Importance')
    pyplt.title('Feature Importance in Random Forest')
    pyplt.show()
def modelTraining(df):
    df.drop(['reservation_status'], axis=1, inplace=True) #We are dropping this because this is the same as is_canceled

    df['is_canceled'] = df['is_canceled'].replace({1: 'Yes', 0: 'No'})

    num = int(input("Enter 1 to use Decision Trees, 2 for Random Forest, 3 for RandomForest with GridSearch, 4 for Random"
                    " Forest with RandomisedSearch, 5 for Logistic Regression, 6 for SVM Classifier "))

    if num == 1:
        decisionTree(df)
    elif num == 2:
        randomForest(df)
    elif num == 3:
        randomForestWithGridSearch(df)
    elif num == 4:
            randomForestWithRandomSearch(df)
    elif num == 5:
            logisticRegression(df)
    elif num == 6:
            svmClassifier(df)
    else:
        print("Enter a number between 1 to 6 only")

    #decisionTree(df)
    #randomForest(df)
    #svmClassifier(df)
    #logisticRegression(df)
    #randomForestWithGridSearch(df)
    #randomForestWithRandomSearch(df)
    #knnModel(df)
    #1l1Regularisation(df)

def encoding(df):
    #Converting object types to Category types
    df['hotel'] = df['hotel'].astype('category')
    df['arrival_date_month'] = df['arrival_date_month'].astype('category')
    df['meal'] = df['meal'].astype('category')
    df['country'] = df['country'].astype('category')
    df['market_segment'] = df['market_segment'].astype('category')
    df['distribution_channel'] = df['distribution_channel'].astype('category')
    df['reserved_room_type'] = df['reserved_room_type'].astype('category')
    df['assigned_room_type'] = df['assigned_room_type'].astype('category')
    df['deposit_type'] = df['deposit_type'].astype('category')
    df['customer_type'] = df['customer_type'].astype('category')
    df['deposit_type'] = df['deposit_type'].astype('category')
    df['reservation_status'] = df['reservation_status'].astype('category')

    df['reservation_status_date'] = df['reservation_status_date'].apply(lambda x: datetime.strptime(x, "%Y-%m-%d").date())
    print(df.dtypes)
    df = df.apply(LabelEncoder().fit_transform)
    print(df.head())
    #Category types: hotel,arrival_date_month,meal,country,market_segment,distribution_channel,reserved_room_type,assigned_room_type,deposit_type,customer_type,reservation_status,

    modelTraining(df)
def dataAnalysis(df):
    #Bar Chart of Cancelled Reservations by Month of Resort Hotel (According to this, highest were in August)
    resort_cancelled_df = df[(df['hotel'] == 'Resort Hotel') & (df['is_canceled'] == 1)]
    cancellations_by_month = resort_cancelled_df.groupby('arrival_date_month')['is_canceled'].sum()
    #print(resort_cancelled_df)

    plt.figure.Figure(figsize=(10, 6))
    cancellations_by_month.plot(kind='bar')
    pyplt.title('Cancelled Bookings in Resort Hotel by Month')
    pyplt.xlabel('Month')
    pyplt.ylabel('Number of Cancelled Bookings')
    pyplt.xticks(rotation=45)
    #pyplt.show()


    #Line Chart Trend of Cancelled Bookings in Resort Hotel by Year
    cancellations_by_month = resort_cancelled_df.groupby('arrival_date_year')['is_canceled'].sum()
    # print(resort_cancelled_df)

    plt.figure.Figure(figsize=(10, 6))
    cancellations_by_month.plot(kind='line')
    pyplt.title('Cancelled Bookings in Resort Hotel by Year')
    pyplt.xlabel('Month')
    pyplt.ylabel('Number of Cancelled Bookings')
    pyplt.xticks(rotation=45)
    #pyplt.show()

    encoding(df)
def handleOutliers(df):
    #Keep Checking between different numerical values
    plt.figure.Figure(figsize=(10, 8))
    pyplt.scatter(df['lead_time'],df['stays_in_week_nights'],s=5)
    pyplt.xlabel('LeadTime')
    pyplt.ylabel('Stay in Week Nights')
    pyplt.title('Scatter Plot of Leadtime vs Stay')
    pyplt.show()

    plt.figure.Figure(figsize=(10, 8))
    pyplt.scatter(df['lead_time'], df['arrival_date_month'], s=5)
    pyplt.xlabel('LeadTime')
    pyplt.ylabel('Arrival Date Month')
    pyplt.title('Scatter Plot of Leadtime vs Arrival Date Month')
    pyplt.show()

    dataAnalysis(df)
def handleDuplicates(df):
    print("**************************************")
    print(df.duplicated().sum())
    #print(df[df.duplicated()])
    df.drop_duplicates(inplace=True)
    print(df.duplicated().sum())

    handleOutliers(df)
def handleNullValues(df):
    #Company contains too many null more than 90% values are null
    #df.info()
    df = df.drop('company',axis=1)
    #print(df['company'])

    #Children has 4 rows with null, vertical fill
    df['children'].fillna(method='ffill', inplace=True)

    #Country has 488 with null, replace with mode
    country_mode = df['country'].mode()[0]

    # Fill null values in the 'country' column with the mode
    df['country'].fillna(country_mode, inplace=True)

    #Agent has 16340 with null, replace with mode
    agent_mode = df['agent'].mode()[0]
    # Fill null values in the 'country' column with the mode
    df['agent'].fillna(agent_mode, inplace=True)

    print(df.isnull().sum())
    #df.info()
    #df.fillna()

    handleDuplicates(df)
def dataCleaning(df):
    print(df.isnull().sum())
    #print(df.duplicated().sum())
    handleNullValues(df)
def dataExploration(df):
    #print(df.head())
    #print(df.shape)
    print(df.info())
    #print(df.describe())
    dataCleaning(df)

def main():
    df =  pd.read_csv('hotel_bookings.csv')
    dataExploration(df)

if __name__ == "__main__":
    main()