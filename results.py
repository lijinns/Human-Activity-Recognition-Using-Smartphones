# -*- coding: utf-8 -*-
"""
@author: Lijin N S
"""

from functions import predict_model
from sklearn.metrics import classification_report

#calling the model
#Available models : 'lr','lsvm','svm','rf','dt'

pred = predict_model(modelname = 'svm', selected_features = 559)

path = #Enter csv file path here
y_true = np.array(pd.read_csv(path,header = None)).reshape(-1)

print(classification_report(y_true, pred))