# Import packages
import re
import nltk
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# Load the dataset
fake_data = pd.read_csv('Fake.csv')
real_data = pd.read_csv('True.csv')

