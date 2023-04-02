import pandas as pd
pd.set_option('display.max_colwidth', None)
import numpy as np
np.set_printoptions(precision=3, suppress=True)
import seaborn as sns
sns.set(style='whitegrid')
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras import regularizers
from sklearn.model_selection import train_test_split
from keras.layers.experimental import preprocessing
import math
import uuid
import random
import zipfile

! pip -q install phe
from phe import paillier

# ! pip -q install clkhash
# from clkhash import clk, randomnames