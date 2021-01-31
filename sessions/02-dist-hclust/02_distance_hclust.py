# -*- coding: utf-8 -*-
"""02-distance-hclust

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1lgQEjft6QX9SvKOINnZUkXafJGAQt1Pt
"""

# installs
# pip install scikit-learn

# imports - usual suspects
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# for distance and h-clustering
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import pdist, squareform


# sklearn does have some functionality too, but mostly a wrapper to scipy
from sklearn.metrics import pairwise_distances 
from sklearn.preprocessing import StandardScaler

# let's start basic
# x = np.array([1,2])
# y = np.array([3,4])
# z = np.array([2,4])
# a = np.stack([x,y,z])
# a_df = pd.DataFrame(a)  # dataframe version
# a                       # nd array

# lets get the euclidean distance


# prints out as pairs 
# 0/1, 0/2, 1/2
# https://stackoverflow.com/a/13079806/155406

# what is it
# technically its a condensed matrix of the upper triangle as 1d array

# but what we are mostly used to is the squareform

# NOTE: there are tools in sklearn, but some methods allow us to pass a compressed matrix
#       which gives us an analysts control over the input space

# QUICK EXERCISE:
#                 calculate the cosine distance matrix
#                 Tricky: calculate the Manhattan distance 
#                       HINT: documentation is your friend

## there are other distance calcs, but I really dont see these come up that often in practical applications
## nothing stopping you from looping parameters to assess what works the best

# we can also use sklearn to calc distances

# QUICK NOTE:
#             some implementations may be faster in sklearn, note the docs

# let's start to code up a simple example

# auth into GCP Big Query

# COLAB Only
# from google.colab import auth
# auth.authenticate_user()
# print('Authenticated')

# for non-Colab
# see resources, as long as token with env var setup properly, below should work

# get the data
SQL = "SELECT * from `questrom.datasets.mtcars`"
YOUR_BILLING_PROJECT = "questrom"

cars = pd.read_gbq(SQL, YOUR_BILLING_PROJECT)

# what do we have?

# the first few rows

# EXERCISE:
#         1) use the model column as the row index 
#         2) with the model column as index, we can now remove it
#         3) explore the data
#   Keep in mind that our goal is to use distance for clustering!
#   Does anything stand out?

# lets drop the model column and use it as the index

# confirm we have what we need

# ok, let's summarise the info

# no missing values is great, finally the summaries

# optional viz, but takes some time
# sns.pairplot(cars)

# keep just the continous variables
# cars2 = pd.concat((cars.loc[:, "mpg"], cars.loc[:, "disp":"qsec"]), axis=1)

# confirm we have what we need

# eventually we want to run the distance matrix through linkage
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html

# use scipy for distance matrix


# cdist = pdist(cars2.values)

## Why?  We have more control, as we could always build our distance matrix to our needs
##       Above is just the mechanics of getting this done

# visualize the matrix with seaborn

# sns.heatmap(squareform(cdist), cmap="Reds")

# Thought exercise:  Why might this help us think about the # of clusters

# lets perform our first hclust!

# hc1 = linkage(cdist)

# now visualize the dendrogram

# dendrogram(hc1, labels=cars.index)
# plt.show()

# the labels for the cluster - cleaner

# dendrogram(hc1,
#            labels = cars.index,
#            leaf_rotation=90,
#            leaf_font_size=10)
# plt.show()

# and the orientation/size

# plt.figure(figsize=(5,6))
# dendrogram(hc1,
#            labels = cars.index,
#            orientation = "left")
# plt.show()

# once we have seen the plots, we can start to think about cutting this up
# to define clusters - we use fcluster
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.fcluster.html

# we can slice up our clusters a few ways
# first, how many clusters (max)

# fcluster(hc1, 2, criterion="maxclust")

# we can also define by the distance

# want to visualize how you defined the cluster?

# DIST = 80
# plt.figure(figsize=(5,6))
# dendrogram(hc1, 
#            labels = cars.index,
#            orientation = "left", 
#            color_threshold = DIST)
# plt.axvline(x=DIST, c='grey', lw=1, linestyle='dashed')
# plt.show()

# YOUR TURN:
#           Use cosine distance
#           generate the linkage array
#           plot the dendrogram
#           assign the cluster labels back onto the ORIGINAL dataframe



# now that we have labels assigned, we can profile

# data dictionary for profiling above
# https://stat.ethz.ch/R-manual/R-devel/library/datasets/html/mtcars.html

# DISCUSSION:
#            This is a simple dataset, but why is profiling important for us as analyts?
#            Applications of this approach?

# ok, 3 more things to consider

# scaling the data to give all features equal importance
# viz different approaches instead of 1x1
# more "advanced" ways to think about distance to help us inform cluster selection

# lets go back to our cars2 dataset (the one with just "continous")

# scaling variables allows each to have equal importance
# since they are now on the same unit scale

# sc = StandardScaler()
# cars_scaled = sc.fit_transform(cars2)

# what do we have

# make it a dataframe

# cars_scaled = pd.DataFrame(cars_scaled, columns=cars2.columns, index=cars2.index)
# cars_scaled.head(3)

# confirm scaled

# METHODS = ['single', 'complete', 'average', 'ward']
# plt.figure(figsize=(15,5))


# # loop and build our plot
# for i, m in enumerate(METHODS):
#   plt.subplot(1, 4, i+1)
#   plt.title(m)
#   dendrogram(linkage(cars_scaled.values, method=m),
#              labels = cars_scaled.index,
#              leaf_rotation=90,
#              leaf_font_size=10)
  
# plt.show()

# I am going to choose ward, choose whatever you like below

# wlink = linkage(cars_scaled.values, method="ward")
# dendrogram(wlink,
#           labels = cars_scaled.index,
#           leaf_rotation=90,
#           leaf_font_size=10)

# plt.show()

# lets look at the distance added at each step
# docs = 4th paragraph for output

# length of the entry

# look at the actual data

# lets look at the growth in distance added

# added_dist = wlink[:, 2]
# added_dist

# calculate the diff at each join

# penalty = np.diff(added_dist)
# penalty[-5:]

# elbow method - what clustering step starts to show signs of explosion in distance
# remember, we lost one via the diff

# sns.lineplot(range(1, len(penalty)+1), penalty)

# we can re-inspect

# set the clusters based on max dist

# labs2 = fcluster(wlink, 5.5, "distance")

# plot it

# dendrogram(wlink,
#           labels = cars_scaled.index,
#           leaf_rotation=90,
#           leaf_font_size=10)
# plt.axhline(y=5.5)
# plt.show()

# ensure intuition aligns with clusters

