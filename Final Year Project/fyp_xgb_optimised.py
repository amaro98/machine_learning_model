import numpy as np
import pandas as pd
import xgboost as xgb
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score,roc_auc_score,make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix
from sklearn.model_selection import StratifiedKFold
from xgboost import plot_tree
'''
[IMPORTING TRAINING DATASET]

-   x : predictors table, we drop STATUS from x
-   y : contains STATUS
-   axis = 1 : drop column, axis = 0 : drop row


'''
df = pd.read_excel('D:\\00 Personal File\\00 ACADEMICS\\4RD YEAR 1ST\FYP PROJECT\WEEK 9\Data_10s_summary_icurve_noitae.xlsx')
#print(df.head()

#DROPPING STATUS COLUMN
x = df.drop('STATUS', axis=1).copy()
y = df['STATUS'].copy()


'''
[Y DATA ENCODING]

-   convert STATUS which is categorical into format readable by machine learning
-   fit_transform(): used on the training data so that we can scale the training data and 
                     also learn the scaling parameters of that data. Here, the model built by us will learn the mean 
                     and variance of the features of the training set. 
                     These learned parameters are then used to scale our test data.
-   LabelEncoder(): Encode target labels with value between 0 and n_classes-1.
                    This transformer should be used to encode target values
'''

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

print(le.classes_)
print(y)


'''
[TRAIN_TEST_SPLIT] 
-   train_test_split:   split data
-   stratify        :   ensure maintain same percentage of status in both training and testing
-   test_size       :   default=None, If float, should be between 0.0 and 1.0 and represent the proportion of the 
                        dataset to include in the test split. If int, represents the absolute number of test samples. 
                        If None, the value is set to the complement of the train size. 
                        If train_size is also None, it will be set to 0.25.

'''

x_train, x_test, y_train, y_test = train_test_split(x, y,random_state= 0, test_size=0.5)

'''
[TRAINING XGBOOST ON TRAINING SET]
-   “multi:softmax” :   set XGBoost to do multiclass classification using the softmax objective, 
                        you also need to set num_class(number of classes)
-   “multi:softprob”;   same as softmax, but output a vector of ndata * nclass, which can be further reshaped to ndata, nclass matrix.
                        The result contains predicted probability of each data point belonging to each class.
-   evel_metric:        Use merrror or mlogloss since we deal with multiple class instead of binary
'''

# ROUND 1
#clf_xgb = xgb.XGBClassifier()
#clf_xgb.fit(x_train,
#            y_train,)


'''
[OPTIMIZATION]
-   use GridSearchCV
-   NOTE : sources states run GridSearchCB sequentially on subsets of parameter options
            rather than all at the same time
-   Consider adjust learning rate to avoid overfitting
-   ROUND 1 : 0.001,0.01,0.3,
-   ROUND 2 : 0.1,0.2,0.3,0.4,0.5
-   ROUND 3 : 0.3,0.31,0.32,0.33,[0.34],0.35


#finding the best range
learning_rate = [0,1,10,100]
gamma = [0.001]
max_depth = [3]
reg_lambda = [8.6]

param_grid = dict(learning_rate=learning_rate,
                  n_estimators=n_estimators,
                  gamma=gamma,
                  max_depth=max_depth,
                  reg_lambda=reg_lambda)
#kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)                               #cross-val at 5 fold
grid_search = GridSearchCV(clf_xgb, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=5)
grid_result = grid_search.fit(x, y)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))
'''

clf_xgb = xgb.XGBClassifier(
    objective='multi:softprob',
    n_estimators=10,
    gamma=0.001,
    learning_rate=0.88,
    max_depth=3,
    reg_lambda=8.6,
    colsample_bytree=0.5
)
clf_xgb.fit(x_train,
            y_train,
            verbose=True,
            early_stopping_rounds=5,
            eval_metric='mlogloss',
            eval_set=[(x_test,y_test)])

from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf_xgb, X=x_train, y=y_train, cv=5)
print("Accuracy: {:.2f} %".format(accuracies.mean()*100))


'''
[PLOTTING CONFUSION MATRIX]
'''

titles_options = [("Confusion Matrix", None),
                    ("Normalised Confusion Matrix", 'true')]

for title, normalize in titles_options:
    disp = plot_confusion_matrix(clf_xgb,
                      x_test,
                      y_test,
                      cmap=plt.cm.Blues,
                      display_labels=["Agressive","Optimised","Sluggish"],
                      normalize=normalize)

    disp.ax_.set_title(title)
    print(title)
    print(confusion_matrix)
plt.show()


'''
[PLOTTING TREE]


bst = clf_xgb.get_booster()
for importance_type in ('weight', 'gain','cover','total_gain','total_cover'):
    print('%s ' %importance_type,bst.get_score(importance_type=importance_type))

node_params = {'shape':'box',
               'style':'filled,rounded',
               'fillcolor' : '#78cbe',}
leaf_params = {'shape':'box',
               'style':'filled',
               'fillcolor' : '#e48038'}

xgb.to_graphviz(clf_xgb,num_trees=0,size="10,10",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)

plot_tree(clf_xgb)
plt.show()
'''