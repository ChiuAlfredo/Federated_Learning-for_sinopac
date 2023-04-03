#%%
#載入套件
import pandas as pd
pd.set_option("display.max_rows", None, "display.max_columns", None)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from keras import preprocessing
from sklearn.model_selection import train_test_split, RepeatedStratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix, precision_recall_curve, auc
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.stats import chi2_contingency
import math
import uuid
import random
import zipfile

# %%
# 下載資料

!wget https://archive.ics.uci.edu/ml/machine-learning-databases/00573/SouthGermanCredit.zip
with zipfile.ZipFile('SouthGermanCredit.zip', 'r') as zip_ref:
    zip_ref.extractall('./SouthGermanCredit/')
# %%
# 清理資料
original_df = pd.read_csv('./SouthGermanCredit/SouthGermanCredit.asc', sep=' ')
original_df.describe()
id = pd.Series(range(0,1000)).apply(lambda i : str(uuid.uuid4()))
df_with_id = original_df.copy()
df_with_id['id'] = id
df_with_id = df_with_id.set_index('id')

client1_data = df_with_id[['laufkont','sparkont','moral','verw','famges','wohn','verm','laufzeit','hoehe','beszeit','kredit']]
client2_data = df_with_id.drop(['laufkont','sparkont','moral','verw','famges','wohn','verm','laufzeit','hoehe','beszeit'], axis=1)

# 切割資料
client1_train, client1_test = train_test_split(client1_data, test_size=0.2, random_state=0)
client2_train, client2_test = train_test_split(client2_data, test_size=0.2, random_state=0)
client1_test['kredit']
def split_x_y(data_train,data_test):
    train_y = data_train["kredit"]
    train_x = data_train.drop("kredit", axis=1)

    test_x = data_test.drop("kredit", axis=1)
    test_y = data_test["kredit"]

    return train_x, train_y, test_x, test_y

client1_train_x,client1_train_y, client1_test_x, client1_test_y = split_x_y(client1_train,client1_test)
client2_train_x,client2_train_y, client2_test_x, client2_test_y = split_x_y(client2_train,client2_test)



common_train_index = client1_train.index.intersection(client2_train.index)
common_test_index = client1_test.index.intersection(client2_test.index)

print(
    'There are {} common entries (out of {}) in client 1 and client 2\'s training datasets,\nand {} common entries (out of {}) in their test datasets'
    .format(
        len(common_train_index),
        len(client1_train),
        len(common_test_index),
        len(client1_test)))
# %%
#vfl
# %%
# 設定參數
batch_size = 32
learning_rate = 1e-3
epochs = 10

# Instantiate an optimizer.
optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=learning_rate)
# Instantiate a loss function.
# Not from logits because of the softmax layer converting logits to probability.
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=False)
# Instantiate a metric function (accuracy)
train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
#%%
# 預測準確率圖
def plot_loss(loss, accuracy):
  plt.plot(loss, label='loss')
  plt.plot(accuracy, label='accuracy')
  plt.xlabel('Epoch')
  # plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

def plot_accuracy(predictions, answers, threshold):
  tp, tn, fp, fn = 0, 0, 0, 0

  for x in range(len(predictions)):
    if answers[x] == 1:
      # if np.argmax(predictions[x]) == 1:
      if predictions[x][1] >= threshold:
        tp = tp + 1
      else:
        fn = fn + 1
    else:
      # if np.argmax(predictions[x]) == 0:
      if predictions[x][1] < threshold:
        tn = tn + 1
      else:
        fp = fp + 1
  
  accuracy = (tp + tn)/(tp + fp + fn + tn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Accuracy: " + str(accuracy))
  print("Precision: " + str(precision))
  print("Recall: " + str(recall))
  # print("Specificity: " + str(specificity))
  print("F-Measure: " + str(2*(recall * precision) / (recall + precision)))

def convert_to_non_sparse(sparse):
  vector_list = np.zeros((len(sparse), 2))
  for x in range(len(sparse)):
    vector_list[x] = [1 - sparse[x], sparse[x]]
  return vector_list
#%%
#client
class Client:

  def __init__(self, train_data_x,train_data_y, test_data_x,test_data_y, labelled):
    # self.__trainX = train_data.copy()
    # self.__testX = test_data.copy()
    # self.labelled = labelled

    # if (labelled):
    #   self.__trainY = self.__trainX.pop('kredit')
    #   self.__testY = self.__testX.pop('kredit')

    # normalizer = tf.keras.layers.Normalization()
    # normalizer.adapt(np.array(self.__trainX.loc[common_train_index]))
    self.__trainX = train_data_x.copy()
    self.__testX = test_data_x.copy()
    self.labelled = labelled
    self.__trainY = train_data_y.copy()
    self.__testY = test_data_y.copy()
    normalizer = tf.keras.layers.Normalization()
    normalizer.adapt(np.array(self.__trainX.loc[common_train_index]))



    self.model = tf.keras.Sequential([
      normalizer,
      layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
      layers.Dropout(0.5),
      layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
      layers.Dropout(0.5),
      layers.Dense(2),
      layers.Softmax()])
    
  def next_batch(self, index):
    self.batchX = self.__trainX.loc[index]
    if not self.labelled:
      grads = []
      self.model_output = np.zeros((len(index), 2))
      for i in range(len(index)):
        with tf.GradientTape() as gt:
          gt.watch(self.model.trainable_weights)
          output_by_example = self.model(self.batchX.iloc[i:i+1], training=True)
          output_for_grad = output_by_example[:,1]
        self.model_output[i] = output_by_example
        grads.append(gt.gradient(output_for_grad, self.model.trainable_weights))
      return grads
    else:
      self.batchY = self.__trainY.loc[index]
      with tf.GradientTape() as self.gt:
        self.gt.watch(self.model.trainable_weights)
        self.model_output = self.model(self.batchX, training=True)
  def cal_model(self):
    return self.model_output
  
  def predict(self, test_index):
    return self.model.predict(self.__testX.loc[test_index])# + 1e-8

  def predict_all(self, index):
    return self.model.predict(pd.concat([self.__trainX, self.__testX]).loc[index])

  def test_answers(self, test_index):
    if self.labelled:
      return self.__testY.loc[test_index]
    
  def test_answers_all(self, index):
    if self.labelled:
      return pd.concat([self.__testY, self.__trainY]).loc[index]
  
  def batch_answers(self):
    if self.labelled:
      return self.batchY

  def loss_and_update(self, a):
    if not self.labelled:
      raise AssertionError("This method can only be called by client 2")
    self.prob = (a + self.model_output)/2
    self.c = self.coefficient_and_update()/len(self.batchX)
    return self.prob, loss_fn(self.batchY, self.prob)
  
  def coefficient_and_update(self):
    if not self.labelled:
      raise AssertionError("This method can only be called by client 2")
    p = self.prob[:,1]
    c = (p-self.batchY)/((p)*(1-p))
    with self.gt:
      output = sum(c * self.model_output[:,1])/len(c)
    grads = self.gt.gradient(output, self.model.trainable_weights)
    optimizer.apply_gradients(zip(grads, self.model.trainable_weights))
    return c
  
  def update_with(self, grads):
    optimizer.apply_gradients(zip(grads, self.model.trainable_weights))

  def assemble_grad(self, partial_grads):
    if not self.labelled:
      raise AssertionError("This method can only be called by client 2")
    # to assemble the gradient for client 1
    for i in range(len(self.c)):
      partial_grads[i] = [x * self.c[i] for x in partial_grads[i]]
    return [sum(x) for x in zip(*partial_grads)]
#%%
# init client
client1 = Client(client1_train_x, client1_train_y,client1_test_x,client1_test_y, False)
client2 = Client(client2_train_x, client2_train_y,client2_test_x,client2_test_y, True)

#%%
common_train_index_list = common_train_index.to_list()
epoch_loss = []
epoch_acc = []

for epoch in range(epochs):
    print(f'run in {epoch} epoch')
    # epoch=0
    random.shuffle(common_train_index_list)
    train_index_batches = [common_train_index_list[i:i + batch_size] for i in range(0, len(common_train_index_list), batch_size)] 
    total_loss = 0.0
    # Iterate over the batches of the dataset.
    for step, batch_index in enumerate(train_index_batches):
        
        partial_grads = client1.next_batch(batch_index)
        client2.next_batch(batch_index)

        prob, loss_value = client2.loss_and_update(client1.cal_model())
        grad = client2.assemble_grad(partial_grads)
        client1.update_with(grad)
        
        total_loss = loss_value + total_loss
        train_acc_metric.update_state(client2.batch_answers(), prob)

    
    train_acc = train_acc_metric.result()
    print(f'-----train accuracy{train_acc}-----')
    train_acc_metric.reset_states()
    epoch_loss.append((total_loss)/(step + 1))
    epoch_acc.append(train_acc)

plot_loss(epoch_loss, epoch_acc)

#%%
client1.predict(common_test_index)
vfl_pred_test = (client1.predict(common_test_index) + client2.predict(common_test_index))/2
vfl_fpr_test, vfl_tpr_test, vfl_thresholds_test = roc_curve(client2.test_answers(common_test_index), vfl_pred_test[:,1])
vfl_gmeans_test = np.sqrt(vfl_tpr_test * (1-vfl_fpr_test))
vfl_ix_test = np.argmax(vfl_gmeans_test)
print('Best Threshold=%f, G-Mean=%.3f\n' % (vfl_thresholds_test[vfl_ix_test], vfl_gmeans_test[vfl_ix_test]))

# predictions and answers are already aligned
plot_accuracy(vfl_pred_test, client2.test_answers(common_test_index), vfl_thresholds_test[vfl_ix_test])
client2.test_answers(common_test_index)

print("AUC: {}".format(roc_auc_score(client2.test_answers(common_test_index), vfl_pred_test[:,1])))
df=pd.DataFrame(client2.test_answers(common_test_index))
vfl_pred_test_label = [1 if p >= vfl_thresholds_test[vfl_ix_test] else 0 for p in  vfl_pred_test[:,1]]
df['predict']=vfl_pred_test_label
df.to_csv('vfl_predict.csv',encoding ='UTF-8-sig')

#evalueate

#%%

# ----------------------------------centralized----------------------------
#%%




# 資料處理
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(client1_train_x))

def plot_loss(history):
  plt.plot(history.history['loss'], label='loss')
  plt.plot(history.history['accuracy'], label='accuracy')
  plt.xlabel('Epoch')
  # plt.ylabel('Error')
  plt.legend()
  plt.grid(True)

def plot_predictions(prediction):
  a = plt.axes(aspect='equal')
  plt.scatter(testY, prediction)
  plt.xlabel('True Values')
  plt.ylabel('Predictions')
  # lims = [500, 1000]
  # plt.xlim([500, 1000])
  # plt.ylim([500, 1000])
  _ = plt.plot

def plot_accuracy(prediction,testY):
  tp = 0
  tn = 0
  fp = 0
  fn = 0
  testY_arr = testY

  for x in range(200):
    # print(str(testY_arr[x]) + "; " + str(prediction[x]))
    # print("predict is : " + str(np.argmax(prediction[x])))
    if testY_arr[x] == 1:
      if np.argmax(prediction[x]) == 1:
        tp = tp + 1
      else:
        fn = fn + 1
    else:
      if np.argmax(prediction[x]) == 0:
        tn = tn + 1
      else:
        fp = fp + 1
  
  accuracy = (tp + tn)/(tp + fp + fn + tn)
  precision = tp / (tp + fp)
  recall = tp / (tp + fn)
  specificity = tn / (tn + fp)
  print("Accuracy: " + str(accuracy))
  print("Precision: " + str(precision))
  print("Recall: " + str(recall))
  # print("Specificity: " + str(specificity))
  print("F-Measure: " + str(2*(recall * precision) / (recall + precision)))

#%% 
# model 
from tensorflow.keras import regularizers

batch_size=32
learning_rate=1e-3

model = tf.keras.Sequential([
      normalizer,
      layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
      layers.Dropout(0.5),
      layers.Dense(128, activation='elu', kernel_regularizer=regularizers.l2(0.01)),
      layers.Dropout(0.5),
      layers.Dense(2),
      layers.Softmax()])

model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])


# fit1


test_results = {}
dnn_history = model.fit(client1_train_x, client1_train_y, epochs=50, verbose=0, batch_size=batch_size)
plot_loss(dnn_history)

#evaluate
test_loss, test_acc = model.evaluate(client1_test_x, client1_test_y, verbose=2)
print('\nTest accuracy:', test_acc)


# result
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
dnn1_predictions = model.predict(client1_test_x)
plot_accuracy(dnn1_predictions,client1_test_y)


cen1_pred_test_label = [1 if p >= vfl_thresholds_test[vfl_ix_test] else 0 for p in  dnn1_predictions[:,1]]
df['predict_cen1']=cen1_pred_test_label
df.to_csv('vfl_cen_predict.csv',encoding ='UTF-8-sig')
#%%
# fit2
normalizer = tf.keras.layers.Normalization()
normalizer.adapt(np.array(client1_train_x))

dnn_history = model.fit(client2_train_x, client2_train_y, epochs=50, verbose=0, batch_size=batch_size)
plot_loss(dnn_history)

#evaluate
test_loss, test_acc = model.evaluate(client2_test_x, client2_test_y, verbose=2)
print('\nTest accuracy:', test_acc)


# result
# probability_model = tf.keras.Sequential([model, tf.keras.layers.Softmax()])
dnn2_predictions = model.predict(client2_test_x)
plot_accuracy(dnn1_predictions,client2_test_y)


cen2_pred_test_label = [1 if p >= vfl_thresholds_test[vfl_ix_test] else 0 for p in  dnn2_predictions[:,1]]
df['predict_cen2']=cen2_pred_test_label
df.to_csv('vfl_cen_predict.csv',encoding ='UTF-8-sig')
