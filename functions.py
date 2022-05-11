# -*- coding: utf-8 -*-
"""
@author: Lijin N S
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold 
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.feature_selection import f_classif,SelectKBest,mutual_info_classif
import warnings

model_dict = {'lr'  : 'Logistic Regression',
              'lsvm': 'Linear Support Vector Machine',
              'svm' : 'Kernel Support Vector Machine',
              'dt'  : 'Decision Tree Classifier',
              'rf'  : 'Random Forest Classifier'}

def get_data(path):
    print('***** Getting the Data ******\n')
    train_df = pd.read_csv(path+"train_data.csv")
    #train data
    trainlabel_df = pd.read_csv(path+"train_labels.csv")
    #train labels
    test_df = pd.read_csv(path+"test_data.csv")
    #test data
    return train_df,trainlabel_df,test_df
  
def clean_data(train_df,trainlabel_df,test_df):
    print('***** Cleaning the Data *****\n')
    mutual_info = mutual_info_classif(train_df,trainlabel_df['Activity'].values.reshape(-1))
    MI_list =[]
    for x in range(len(mutual_info)):
        MI_list.append([train_df.columns[x],mutual_info[x]])
        MI_dict = dict(MI_list)
    MI_dict_sorted = dict(sorted(MI_dict.items(), key=lambda item: item[1]))
    train_df = train_df.drop([ 'tBodyGyroJerk-arCoeff()-Y,4','subject','tBodyGyroJerkMag-arCoeff()3' ],axis = 1)
    test_df = test_df.drop([ 'tBodyGyroJerk-arCoeff()-Y,4','subject','tBodyGyroJerkMag-arCoeff()3' ],axis = 1)
    for key in [ 'tBodyGyroJerk-arCoeff()-Y,4','subject','tBodyGyroJerkMag-arCoeff()3' ]:
        MI_dict_sorted.pop(key)
    sorted_features = list(MI_dict_sorted.keys())[::-1]
    train_df = train_df[sorted_features]
    test_df = test_df[sorted_features]
    return train_df,test_df
  
# defines the parameter tuning guidelines
def param_tuning(model,params,num):
    k = num 
    kf = KFold(n_splits=k, random_state=1, shuffle = True)
    grid_search = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=kf, scoring='f1_micro',error_score=0)
    return grid_search

# calls the model and defines the parameters to be tuned for each model.
def classifier(modelname):
    if modelname == 'lr':
        model = LogisticRegression()
        params = {
            'penalty' : ['l1', 'l2'],
            'C': np.arange(10,61,10)}
    elif modelname == 'lsvm':
        model = LinearSVC(tol = 0.00005)
        params = {
            'C': np.arange(1,12,2),}
    elif modelname == 'svm':
        model = SVC(kernel = 'rbf')
        params = {
            'C':[2,4,8,16], 
            'gamma':[0.125, 0.250, 0.5, 1]}
    elif modelname == 'dt':
        model = DecisionTreeClassifier()
        params = {
            'max_depth':np.arange(2,10,2)}
    elif modelname == 'rf':
        model = RandomForestClassifier()
        params = {
            'n_estimators': np.arange(20,101,10), 
            'max_depth':np.arange(2,16,2)}
    return model,params

# defines model, parameters to be tuned and parameter tuning guidelines by model name.
def return_model(modelname):
    model,params = classifier(modelname)
    grid_search = param_tuning(model,params,10)
    return grid_search

# returns the train_data set with N best features
def feature_selection(train_df,selected_features):
    new_train_df = train_df.iloc[:,:selected_features]
    return new_train_df

# returns precision, recall, accuracy and f-measure values
def report(y_test,pred):
    accuracy = accuracy_score(y_test, pred)
    fmeasure = f1_score(y_test,pred,average = 'macro')
    precision = precision_score(y_test,pred,average = 'macro')
    recall = recall_score(y_test,pred,average = 'macro')
    print("Accuracy :", accuracy,'\n')
    #print("Precision :", precision,'\n')
    #print("Recall :", recall,'\n')
    #print("F-measure :", fmeasure,'\n')
    report = classification_report(y_test, pred,digits = 4)
    print(report,'\n\n')
    return accuracy,precision,recall,fmeasure  

path = r'Enter path here'
train_df,trainlabel_df,test_df = get_data(path)
train_df,test_df = clean_data(train_df,trainlabel_df,test_df)

# main function
def fit_model(modelname,selected_features = 559):
    print('Model : {}, Selected Features: {}\n'.format(model_dict[modelname],selected_features))
    print('***** Fitting the Model *****\n')
    grid_search = return_model(modelname)
    new_train_df = feature_selection(train_df,selected_features)
    X_train, X_test, y_train, y_test = train_test_split(new_train_df, trainlabel_df['Activity'], test_size=0.2, random_state=42)
    grid_search.fit(X_train, y_train.values.reshape(-1))
    pred = grid_search.predict(X_test)
    accuracy,precision,recall,fmeasure = report(y_test,pred)
    return accuracy,precision,recall,fmeasure
