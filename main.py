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



# 資料切割
x_train, y_train, x_test = get_datasets(
    pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
)

n_data = x_train.shape[0]

# 欄位切割
split_train = partition_data(x_train)
split_test = partition_data(x_test)

outputs = dict()
accuracies = dict()
models = dict()

for i, (_train, _test) in enumerate(zip(split_train, split_test)):
    _train, _test = scale(_train, _test)
    _model = LogisticRegression()
    _model.fit(_train, y_train)

    outputs[i] = _model.predict_proba(_train)
    accuracies[i] = 100 * accuracy_score(_model.predict(_train), y_train)
    models[i] = _model

# ----- Combined data stage -----
train_combined = np.empty((n_data, 0))

for _train in outputs.values():
    train_combined = np.hstack((train_combined, _train))

comp_server = MLPClassifier(
    hidden_layer_sizes=(500, 500,), learning_rate_init=0.0001, verbose=True
)
comp_server.fit(train_combined, y_train)

pred_train_combined = comp_server.predict(train_combined)
# pred_test_combined = comp_server.predict(test_combined)

train_acc_combined = accuracy_score(pred_train_combined, y_train)

for i, acc in accuracies.items():
    print(f"Holder {i} train accuracy: {acc:.3f}%")

print(f"\033[31m\tdecentralized Combined accuracy: {100*train_acc_combined:.3f}%\033[0m")

#%%
# centralized
"""
Train a LogisticRegression model
on the Titanic dataset
"""

import sys
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.feature_engineering import get_datasets, partition_data, scale



x_train, y_train, x_test = get_datasets(
    pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
)
x_train, x_test = scale(x_train, x_test)

lr = LogisticRegression()
lr.fit(x_train, y_train)

pred_train = lr.predict(x_train)
pred_test = lr.predict(x_test)

train_acc = accuracy_score(pred_train, y_train)
print(f"\033[31m\tcentralized Train accuracy: {100*train_acc:.3f}%\033[0m")

# %%
