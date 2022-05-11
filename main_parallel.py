# -*- coding: utf-8 -*-
"""
@author: Lijin N S
"""

from functions import fit_model
import pandas as pd
import argparse

parser = argparse.ArgumentParser(description='Enter model name and number of features.')
parser.add_argument('--model', metavar='M', type=str, default='svm', help='model name')
parser.add_argument('--features', metavar='F', type=int, default=559, help='model name')
args = parser.parse_args()

# Available models : 'lr','lsvm','svm','rf','dt'
# Feature range : (1,559)

accuracy,precision,recall,fmeasure = fit_model(modelname = args.model, selected_features = args.features)