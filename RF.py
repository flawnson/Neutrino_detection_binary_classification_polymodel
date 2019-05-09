"""RANDOM FOREST"""
import pydotplus
import lepton

from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.externals.six import StringIO  
from IPython.display import Image  

"""IMPORT DATA""" 

data = lepton.processed_datasets.pop()

X_data_train = lepton.X_data_pairs.pop()[0]
X_data_test = lepton.X_data_pairs.pop()[1]

Y_data_train = lepton.Y_data_pairs.pop()[0]
Y_data_test = lepton.Y_data_pairs.pop()[1]

train_loader = zip(X_data_train, Y_data_train)

"""MODEL"""

dot_data = StringIO()
export_graphviz(clf, out_file=dot_data,  
                filled=True, rounded=True,
                special_characters=True,feature_names = feature_cols,class_names=['0','1'])
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())  
graph.write_png('diabetes.png')
Image(graph.create_png())