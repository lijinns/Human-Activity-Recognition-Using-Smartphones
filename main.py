# -*- coding: utf-8 -*-
"""
@author: Lijin N S
"""

from functions import fit_model
import pandas as pd

model_dict = {'lr'  : 'Logistic Regression',
              'lsvm': 'Linear Support Vector Machine',
              'svm' : 'Kernel Support Vector Machine',
              'dt'  : 'Decision Tree Classifier',
              'rf'  : 'Random Forest Classifier'}
training_results = []

models = ['lr','lsvm','svm','dt','rf']
features = [100,200,300,400,500,559]

for model in models:
    for feature in features:
        accuracy,precision,recall,fmeasure = fit_model(modelname = model, selected_features = feature)
        training_results.append([model_dict[model],feature,precision,recall,fmeasure])
        
results = pd.DataFrame(training_results)
results.columns =['Model','Features','Precision','Recall','Fmeasure']

results.sort_values('Fmeasure',ascending = False)

print(results)