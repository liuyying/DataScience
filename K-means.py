import numpy as np
from matplotlib import cm, pyplot as plt
import matplotlib.dates as dates
import pandas as pd
import datetime

from scipy import stats # To perfrom box-cox transformation
from sklearn import preprocessing # To center and standardize the data.
from sklearn import cluster # To apply cluster analysis
from operator import truediv # To do the vector division

# Validate the normal distribution of return rate.

# Use open, close, high, low, alpha#23 and alpha#101 to do the cluster analysis

users=pd.read_csv("data/users.csv")
user_id = users["user_id"]
mean_count = users.values[:,:6]
k_means = cluster.KMeans(n_clusters=2, max_iter=5000)
# Train the cluster analysis model
k_means.fit(mean_count)

# Do the in-sample prediction
test_labels = k_means.predict(mean_count)

lei1 = []

lei2 = []

test_cluster_1_return_rate = []

test_cluster_2_return_rate = []

# For the training data histogram.

for i in range(0, len(user_id)):

    if k_means.labels_[i] == 1:
        lei1.append(user_id[i])

    else:
        lei2.append(user_id[i])

