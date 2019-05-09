import pandas as pd  
import numpy as np  
import matplotlib.pyplot as plt
import lepton

from sklearn.svm import SVC

"""IMPORT DATA""" 

data = lepton.processed_datasets.pop()

X_data_train = lepton.X_data_pairs.pop()[0]
X_data_test = lepton.X_data_pairs.pop()[1]

Y_data_train = lepton.Y_data_pairs.pop()[0]
Y_data_test = lepton.Y_data_pairs.pop()[1]

train_loader = zip(X_data_train, Y_data_train)

"""MODEL"""

svclassifier = SVC(kernel='linear')  
svclassifier.fit(X_train, y_train)  