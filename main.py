"""
Run a (local) vertical federated learning
process on Titanic dataset using
LogisticRegression models
"""

# vertical federated learning 
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.feature_engineering import get_datasets, partition_data, scale
from sklearn.ensemble import RandomForestClassifier


# 資料切割
x_train, y_train, x_test, y_test = get_datasets(
    pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
)

n_data_train = x_train.shape[0]
n_data_test = x_test.shape[0]

# 欄位切割
split_train = partition_data(x_train)
split_test = partition_data(x_test)


models = dict()

outputs_train = dict()
accuracies_train = dict()

outputs_test = dict()
accuracies_test = dict()


for i, (_train, _test) in enumerate(zip(split_train, split_test)):
    _train, _test = scale(_train, _test)
    _model = LogisticRegression()
    _model.fit(_train, y_train)

    outputs_train[i] = _model.predict_proba(_train)
    accuracies_train[i] = 100 * accuracy_score(_model.predict(_train), y_train)
    models[i] = _model

    outputs_test[i] = _model.predict_proba(_test)
    accuracies_test[i] = 100 * accuracy_score(_model.predict(_test), y_test)



# ----- Combined data stage -----
## train_score
train_combined = np.empty((n_data_train, 0))

for _train in outputs_train.values():
    train_combined = np.hstack((train_combined, _train))

# # mlp
# comp_server = MLPClassifier(
#     hidden_layer_sizes=(500, 500,), learning_rate_init=0.001, verbose=True
# )

# rf
comp_server = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

comp_server.fit(train_combined, y_train)

pred_train_combined = comp_server.predict(train_combined)

train_acc_combined = accuracy_score(pred_train_combined, y_train)


##test_score

test_combined = np.empty((n_data_test, 0))

for _test in outputs_test.values():
    test_combined = np.hstack((test_combined, _test))

# # mlp
# comp_server = MLPClassifier(
#     hidden_layer_sizes=(500, 500,), learning_rate_init=0.001, verbose=True
# )

# rf
# comp_server = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)

# comp_server.fit(test_combined, y_test)

pred_test_combined = comp_server.predict(test_combined)

test_acc_combined = accuracy_score(pred_test_combined, y_test)


# score
for i, acc in accuracies_train.items():
    print(f"\033[31m\tHolder {i} train accuracy: {acc:.3f}%\033[0m")

print(f"\033[31m\tdecentralized Combined_mlp train accuracy: {100*train_acc_combined:.3f}%\033[0m")

for i, acc in accuracies_test.items():
    print(f"\033[31m\tHolder {i} test accuracy: {acc:.3f}%\033[0m")

print(f"\033[31m\tdecentralized Combined_mlp test accuracy: {100*test_acc_combined:.3f}%\033[0m")



#%%
# centralized
"""
Train a LogisticRegression model
on the Titanic dataset
"""

x_train, x_test = scale(x_train, x_test)

lr = LogisticRegression()
lr.fit(x_train, y_train)

pred_train = lr.predict(x_train)
pred_test = lr.predict(x_test)

train_acc = accuracy_score(pred_train, y_train)
test_acc = accuracy_score(pred_test, y_test)
print(f"\033[31m\tcentralized Train accuracy: {100*train_acc:.3f}%\033[0m")
print(f"\033[31m\tcentralized Test accuracy: {100*test_acc:.3f}%\033[0m")


# %%
