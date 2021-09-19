
 <h1> MACHINE LEARNING BASED TOOL FOR PID CONTROLLER TUNING STATUS </h1>
Disclaimer: This Readme is mainly about developing XGBoost model in ptyhon, the other models were developed using MATLAB Classification Learner<br>
<h2>OVERVIEW</h2><br><br>
<b>Aim: To apply Machine Learning Model to predict the PID Controller Tuning Status</b><br>

The flow of the methodology are illustrated below:<br><br>

![image](https://user-images.githubusercontent.com/88897287/133895401-636c1ba4-0b54-4766-960a-383595ff2af1.png)


<h2>EXECUTION</h2>
<h3>Importing Data</h3>
The first step in working with machine learning is to import the training data generated previously. For XGBoost, the data is stored in an Excel file thus the command “read_excel” is applied from the “pandas” module. The source code below is the command made in python to import training data <br><br>

```
import pandas as pd
dataset = pd.read_excel('Data_10s_summary_icurve_noitae.xlsx')
```

<h3>Data Preprocessing</h3>
* Extracting features<br><br>

![image](https://user-images.githubusercontent.com/88897287/133895509-e7c659cb-0354-4bb8-a1de-5f8c7c118b7e.png)
<br>

```
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
```

The next step is to ensure that the dependent variables are readable by the machine learning model. The main issue in this project is the response column which consists of the categorical dataset which is considered unreadable by most machine learning models. To solve this, data encoding is performed.

```

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_train = le.fit_transform(y_train)

from tabulate import tabulate
y_test = le.fit_transform(y_test)
```
The illustration of Data Encoding: <br><br>
![image](https://user-images.githubusercontent.com/88897287/133895538-d81490c0-30b8-489f-a7dc-d5ad5ff0d13f.png)

<h3>Developing Base Model</h3>

```
from xgboost import XGBClassifier
classifier = XGBClassifier()
classifier.fit(X_train, y_train)
```

Next step is to predict test set
```
y_pred = classifier.predict(X_test)
```

Then the accuracy is obtained, we proceed with Hyperparameter tuning to see whether it can perform better

<h3>Hyperparameter Tuning using GridSearchCV</h3>

```
%Finding the best learning rate:
learning_rate = [0,1,10,100]
#finding the best range
learning_rate = [0,1,10,100]

param_grid = dict(learning_rate=learning_rate)
grid_search = GridSearchCV(classifier, param_grid, scoring="neg_log_loss", n_jobs=-1, cv=5)
grid_result = grid_search.fit(x_train, y_train)

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
print("%f (%f) with: %r" % (mean, stdev, param))
```
The best hyperparameter value will be shown based on the lowest log loss. The process is repeated to obtain a more precise value and to find the other best hyperparameter value.

The finalised XGBoost Model:

```
from xgboost import XGBClassifier
classifier = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0.001,
              learning_rate=1, max_delta_step=5, max_depth=3, min_child_weight=1, missing=None, n_estimators=150, n_jobs=1,
              nthread=None, objective='multi:softprob', random_state=0, reg_alpha=0, reg_lambda=8.6, scale_pos_weight=1,
              seed=42, silent=None, subsample=1, verbosity=1)
```


<h3>Result</h3>

![image](https://user-images.githubusercontent.com/88897287/133895634-43b56260-648f-4c7e-ad0a-35aa42d51e9b.png)

Train Set Accuracy : 99.2%

![image](https://user-images.githubusercontent.com/88897287/133895648-4cc9ddd9-7261-417b-a5d9-071203f35af5.png)

Test Set Accuracy : 98.40%

<h2>COMPARISON BETWEEN OTHER MACHINE LEARNING MODELS</h2>

![image](https://user-images.githubusercontent.com/88897287/133916491-75ab8467-e2c8-4ba3-8e5a-2f9c58491e65.png) <br><br>
Con't<br>
![image](https://user-images.githubusercontent.com/88897287/133916487-5afc2e51-ca6c-4bd8-b1c6-bef599d45ef9.png)




