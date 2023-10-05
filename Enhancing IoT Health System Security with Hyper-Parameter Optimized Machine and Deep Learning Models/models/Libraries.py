#Import Libraries/Packages/Dependencies:
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from scipy import stats
import os
import random
from tqdm import tqdm
import logging
import itertools
import time
import math
from collections import defaultdict
from datetime import timedelta
from datetime import date
import copy

import IPython
import IPython.display
from IPython.display import display

from sklearn import preprocessing
from sklearn.preprocessing import scale
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score, auc, roc_curve, confusion_matrix, classification_report, precision_recall_fscore_support, f1_score
from sklearn.metrics import accuracy_score, auc, f1_score, precision_score, recall_score, roc_curve
from sklearn.utils import shuffle
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn import tree
from sklearn.multiclass import OneVsOneClassifier, OneVsRestClassifier
from collections import Counter

from keras.models import load_model
from keras.utils.vis_utils import plot_model
from keras.models import Model
import tqdm, re, sys

from keras.layers import Conv2D, Conv1D, MaxPooling2D, MaxPooling1D, Flatten, BatchNormalization, Dense, LSTM, Dropout, Bidirectional
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.callbacks import CSVLogger, ModelCheckpoint
from keras.layers import Layer, SpatialDropout1D

!pip install -q -U keras-tuner
import kerastuner as kt
import keras_tuner
from kerastuner.tuners import RandomSearch

from keras import backend as K

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

torch.backends.cudnn.benchmark=True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.manual_seed(5703)
torch.manual_seed(5703)
np.random.seed(5703)
random.seed(5703)

!pip install -U -q PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

# Authenticate and create the PyDrive client.
auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

!pip install opendatasets
import opendatasets as od

import requests
import csv
from io import StringIO

from scipy.signal import resample
from sklearn.utils import resample
import imblearn
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import BorderlineSMOTE

! pip install cloudstor
from cloudstor import cloudstor

import warnings
warnings.filterwarnings('ignore')

## Load Datasets for the IoMT Devices:
from google.colab import drive
drive.mount('/content/drive')
