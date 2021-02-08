##############################################################################
## Dimension Reduction 1: Principal Components Analysis
## Learning goals:
## - application of PCA in python via sklearn
## - data considerations and assessment of fit
## - hands on data challenge = Put all of your skills from all courses together!
## - setup non-linear discussion for next session
##
##############################################################################

# installs

# notebook/colab
#

# local/server
pip install wheel
pip install pandas
pip install scikit-plot

# imports

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# what we need for today
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import metrics 

import scikitplot as skplt

# color maps
from matplotlib import cm

# warmup exercise
# questrom.datasets.diamonds
# 1. write SQL to get the diamonds table from Big Query
# 2. keep only numeric columns (pandas can be your friend here!)
# 3. use kmeans to fit a 5 cluster solution
# 4. generate the silohouette plot for the solution
# 5. create a boxplot of the column carat by cluster label (one boxplot for each cluster)


# from google.colab import auth
# auth.authenticate_user()
# print('Authenticated')

##################################
## PCA
##################################

# get the judges data from Big Query
# we will use the same dataset as last week to start
# questrom.datasets.judges

# write a sql statement to get the judges dataset

# shape

# columns

# cleanup

# judges.index = judges.judge
# del judges['judge']

# first few rows to confirm

# lets review correlation

## QUESTION:  2 minutes, review the judges correlation matrix
##            if one way to reduce variables is to reduce via
##            correlation, what would you select for variables?

# lets fit our first model
# fit transform fits the PCA model AND applies 
# pca object has the fit data we will explore, pcs are the new features

# pca = PCA()
# pcs = pca.fit_transform(judges)

# what do we have?

# shape confirmation (rows/features) are identical SHAPES

# pcs.shape == judges.shape

###################################
# the core things we are looking for are the components (the new features)
# the explained variance ratio for each, to help determine the cutoff
# inverse_transform to put the data back into original  space
###################################

# first, lets get the explained variance
# elbow plot

# varexp = pca.explained_variance_ratio_
# sns.lineplot(range(1, len(varexp)+1), varexp)

# cumulative variance

# plt.title("Cumulative Explained Variance")
# plt.plot(range(1, len(varexp)+1), np.cumsum(varexp))
# plt.axhline(.95)

## EXERCISE:  Take 7 minutes:
##            use the same diamonds dataset as the warmup exercise (from Big Query)
##            do you have to do any data cleaning?
##            EXCLUDE the price variable, we will use it later
##            review the correlation matrix
##            STANDARDIZE the data
##            fit a PCA model 
##            generate plots to review how many PCs we might retain

##### Return to the judges dataset

# quick function to construct the barplot easily
def ev_plot(ev):
  y = list(ev)
  x = list(range(1,len(ev)+1))
  return x, y

# going back to judges PCAs object pca
# another approach for selection is to use explained variance > 1

# x, y = ev_plot(pca.explained_variance_)
# sns.barplot(x, y)

## ok, lets step back
## 1.  PCA we fit did not specify any settings, we will explore pre-config setup in a moment
## 2.  Once we fit (.fit), the object has some useful information for us to explore
## 3.  We extracted the components (new features) with .transform
## 4.  We can use the summary info to inform our decisions as an analyst

## But lets dive even deeper

# original variable contributes to the PC construction
# these are generally referred to as loadings
# good resource: https://scentellegher.github.io/machine-learning/2020/01/27/pca-loadings-sklearn.html

# component, feature


# build column labels

# COLS = ["PC" + str(i) for i in range(1, len(comps)+1)]
# loadings = pd.DataFrame(comps.T, columns=COLS, index=judges.columns)
# loadings

## so what do we have?

# remember we talked about the linear combination?
# these are the weights  contribution) used to construct the components from the original variables
# TIP:  these are referred to as rotations in the output in R

# What I tend to do is use this to try to explain how variables are contributing

# help with hacking on matplotlib
# https://matplotlib.org/3.1.1/gallery/images_contours_and_fields/image_annotated_heatmap.html

## 2 Minute thought exercise:
##     Recall from earlier plots the variance explained and correlation
##     Does this make sense?  Why or why not?

## Quick hands-on exercise:
##     go back to your diamonds pca
##     review the contribution of each variable on each PC
##     what do you see?

# moving forward, application 1 is to use these new feautres instead of the oringal features

# quick refresher, review pcs

# shape

# slice all "rows", and the first 2 components

# first few rows and columns

# make a new dataframe
# remember, these are what we might use if our task was to learn a model

# j = pd.DataFrame(comps, columns=['pc1', 'pc2'], index=judges.index)
# j.head()

## notice that I am NOT putting these back onto the original
## you can of course, but the point is that these are now our new features for any other downstream tasks

# viz

## Thought exercise:
##       what do you see in the plot above?  Anything
##       what might you want to do now?

## Quick exercise:
##       using the two pcs from judges
##       fit keans = 2
##       plot the same scatter plot from above, but color the points by cluster assignment
##       What do you think about the cluster assignments?
##       What might you do to improve?

#############################
## some options when fitting PCA Model
## we can pre-specify the number of components OR the % of var we want to explain!
## lets use diamonds

# lets rebuild to practice and show some quick 1-liners

# arbitrary, but lets keep just 2 components
# using svd_solver full to keep solutions consistent
# scikit tries to be smart with auto, but I want to the same solver for examples

# variance explained in just 2?

# I will violate my rule and put back onto the original just for exploration

# dia[['pc2_1', 'pc2_2']] = dia_pc2
# dia.head(3)

# plot it up
# https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html

# dia['cut2'] = dia.cut.astype('category').cat.codes
# plt.scatter(x=dia.pc2_1, y=dia.pc2_2, c=dia.cut2, cmap=cm.Paired, alpha=.3)

# lets fit another PCA, but keep 90% variance

# Violate again, see results

# now lets look at everything on the original dataframe

##################################################################
##
##   Compression
## 
## once we have fit our model, we have seen that transform genera†es the newly constructed features
## but we can also project back into the original feature space

# quick refresher, what was the shape that we used?
# diamonds pcs we just fit, 2 and 90

# lets use 90% (3 components)

# put back into a dataframe
# remember, I scaled dia_n -> dia_s

# comp3df = pd.DataFrame(comp3, columns=dia_n.columns)
# comp3df.head(3)



# remember the MNIST dataset?
# I showed the aspect of compression there, a simple link
# the digit is reconstructed after fitting N components and .inverse_tranform
# depends on our use case of course, but highlights that "losing" information may not hurt your applications
# depending on the dataset, our problem, and our method, we can potentially remove noise and only retain signal!

# https://snipboard.io/qrohR7.jpg

##################################################################
##
##   Next steps
## 
## - Diamonds data challenge in breakout rooms
## - lets start to see how we can combine UML and SML!
##
## - OBJECTIVE:  As a group, fit a regression model to the price column
## -             What is your R2? can you beat your best score?
## 
##
## 1. refit PCA to the diamonds dataset.
## 2. how many components would you select
## 3. remember!  do not include price in your columns when generating the components, we are predicint that!
## 4. Iterate!  try different models, assumptions
##
## NOTE:  we haven't covered regression in scikit learn, but its the same flow!
## Some help:  
##   
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score 
##################################################################
