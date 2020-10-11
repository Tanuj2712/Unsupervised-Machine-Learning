import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import plotly.graph_objs as go
import plotly .offline as offline
import plotly.figure_factory as ff
import statistics


# Importing dataset and examining it
dataset = pd.read_csv("Drink.csv")
print(dataset.head())
print(dataset.shape)
print(dataset.info())
print(dataset.describe())

# Plotting Correlation Heatmap
#corrs = dataset.corr()
#figure = ff.create_annotated_heatmap(
#    z=corrs.values,
#    x=list(corrs.columns),
#    y=list(corrs.index),
#    annotation_text=corrs.round(2).values,
#    showscale=True)
#offline.plot(figure,filename='corrheatmap.html')

## Dropping columns with high correlation + causation
X = dataset.drop('density', axis=1)
#print(type(X))
#print(X.shape)

# Dividing data into subsets
#Acidic Data
subset1 = dataset[['fixed acidity','volatile acidity','citric acid','pH']]
print(subset1.describe())
z = statistics.median(subset1['volatile acidity'])
print(z)

def converterFixed(column):
    if column <= 6.85:
        return 0 # Low Fixed Acidity
    else:
        return 1 # High Fixed Acidity

X['fixed acidity'] = X['fixed acidity'].apply(converterFixed)


def converterpH(column):
    if column <= 3.18:
        return 0 # Low pH 
    else:
        return 1 # High pH

X['pH'] = X['pH'].apply(converterpH)

def converterVolatile(column):
    if column <= .26:
        return 0 # Low Volatile 
    else:
        return 1 # High Volatile

X['volatile acidity'] = X['volatile acidity'].apply(converterVolatile)

#fig, ax = plt.subplots(2, 2, figsize=(12,10))
#ax[0, 0].hist(subset1['fixed acidity'],color='#DC4405',bins= 62,alpha=0.5, edgecolor='black')
#ax[0, 0].title.set_text('Fixed Acidity')
#ax[0, 1].hist(subset1['volatile acidity'],color='#DC4405',bins= 62,alpha=0.5, edgecolor='black')
#ax[0, 1].title.set_text('Volatile Acidity')
#ax[1, 0].hist(subset1['citric acid'],color='#DC4405',alpha=0.5, edgecolor='black')
#ax[1, 0].title.set_text('Citric Acidity')
#ax[1, 1].hist(subset1['pH'],color='#DC4405',alpha=0.5, edgecolor='black')
#ax[1, 1].title.set_text('pH Level')
#plt.show()



##Taste and Density Data
subset2 = X[['residual sugar','chlorides','alcohol']]
def converterTaste(column):
    if column <= 6.4:
        return 0 # Low Residual Sugar
    else:
        return 1 # High Residual Sugar
dataset['residual sugar'] = dataset['residual sugar'].apply(converterTaste)

def converterAlcohol(column):
    if column <= 10.5:
        return 0 # Low Alcohol Percent
    else:
        return 1 # High Alcohol Percent
dataset['alcohol'] = dataset['alcohol'].apply(converterAlcohol)


#fig, ax = plt.subplots(2, 2, figsize=(12,10))
#ax[0, 0].hist(subset2['residual sugar'],color='#DC4405',bins= 62,alpha=0.5, edgecolor='black')
#ax[0, 0].title.set_text('Residual Sugar')
#ax[0, 1].hist(subset2['chlorides'],color='#DC4405',bins= 62,alpha=0.5, edgecolor='black')
#ax[0, 1].title.set_text('Chlorides')
#ax[1, 0].hist(subset2['density'],color='#DC4405',alpha=0.5, edgecolor='black')
#ax[1, 0].title.set_text('Density')
#ax[1, 1].hist(subset2['alcohol'],color='#DC4405',alpha=0.5, edgecolor='black')
#ax[1, 1].title.set_text('Alcohol Percent')
#plt.show()

# Sulfur Data
subset3 = dataset[['free sulfur dioxide', 'total sulfur dioxide', 'sulphates']]
print(subset3.describe())
#fig, ax = plt.subplots(1, 3, figsize=(12,10))
#ax[0].hist(subset3['free sulfur dioxide'],color='#DC4405',bins= 62,alpha=0.5, edgecolor='black')
#ax[0].title.set_text('Free Sulfur')
#ax[1].hist(subset3['total sulfur dioxide'],color='#DC4405',bins= 62,alpha=0.5, edgecolor='black')
#ax[1].title.set_text('Total Sulfur')
#ax[2].hist(subset3['sulphates'],color='#DC4405',alpha=0.5, edgecolor='black')
#ax[2].title.set_text('Sulphates')
#plt.show()

def converterTotalSulfur(column):
    if column <= 138.36:
        return 0 # Low level free sulfur
    else:
        return 1 # High level free sulfur
dataset['total sulfur dioxide'] = dataset['total sulfur dioxide'].apply(converterTotalSulfur)

#Combining important factors
subset4 = dataset[['free sulfur dioxide', 'total sulfur dioxide', 'residual sugar', 'alcohol','fixed acidity','volatile acidity']]

# Normalizing numerical features so that each feature has mean 0 and variance 1
feature_scaler = StandardScaler()
X1 = feature_scaler.fit_transform(subset1)
X2 = feature_scaler.fit_transform(subset2)
#X3 = feature_scaler.fit_transform(subset3)
#X4 = feature_scaler.fit_transform(subset4)

# Analysis on Acidic Data
# Finding the number of clusters (K) - Elbow Plot Method
#inertia = []
#for i in range(1,11):
#    kmeans = KMeans(n_clusters = i, random_state = 100)
#    kmeans.fit(X1)
#    inertia.append(kmeans.inertia_)

#plt.plot(range(1, 11), inertia)
#plt.title('The Elbow Plot')
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#plt.show()

## Running KMeans to generate labels
#kmeans = KMeans(n_clusters = 2)
#kmeans.fit(X1)



## Implementing t-SNE to visualize dataset
#tsne = TSNE(n_components = 2, perplexity =45,n_iter=3000)
#x_tsne = tsne.fit_transform(X1)

#fixed = list(X['fixed acidity'])
#volatile = list(X['volatile acidity'])
#citric = list(X['citric acid'])
#pH = list(X['pH'])
#data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
#                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
#                                text=[f'Fixed: {a}; Volatile: {b}; Citric:{c}, pH:{d}' for a,b,c,d in list(zip(fixed,volatile,citric,pH))],
#                                hoverinfo='text')]

#layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 800,
#                    xaxis = dict(title='First Dimension'),
#                    yaxis = dict(title='Second Dimension'))
#fig = go.Figure(data=data, layout=layout)
#offline.plot(fig,filename='t-SNE.html')


# Analysis on Taste and Density Data
# Finding the number of clusters (K) - Elbow Plot Method
inertia = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i, random_state = 100)
    kmeans.fit(X2)
    inertia.append(kmeans.inertia_)

plt.plot(range(1, 11), inertia)
plt.title('The Elbow Plot')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.show()

## Running KMeans to generate labels
#kmeans = KMeans(n_clusters = 2)
#kmeans.fit(X2)

##Implementing t-SNE to visualize dataset
#tsne = TSNE(n_components = 2, perplexity =40 ,n_iter=3000)
#x_tsne = tsne.fit_transform(X2)

#sugar = list(dataset['residual sugar'])
#chlorides = list(dataset['chlorides'])
##density = list(dataset['density'])
#alcohol = list(dataset['alcohol'])

#data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
#                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
#                                text=[f'Sugar: {a}; Chlorides: {b};  Alcohol:{c}' for a,b,c in list(zip(sugar,chlorides,alcohol))],
#                                hoverinfo='text')]

#layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
#                    xaxis = dict(title='First Dimension'),
#                    yaxis = dict(title='Second Dimension'))
#fig = go.Figure(data=data, layout=layout)
#offline.plot(fig,filename='t-SNE2.html')

# Analysis on Sulfurs and Sulphates data
# Finding the number of clusters (K) - Elbow Plot Method
#inertia = []
#for i in range(1,11):
#    kmeans = KMeans(n_clusters = i, random_state = 100)
#    kmeans.fit(X3)
#    inertia.append(kmeans.inertia_)

#plt.plot(range(1, 11), inertia)
#plt.title('The Elbow Plot')
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#plt.show()

# Running KMeans to generate labels
#kmeans = KMeans(n_clusters = 2)
#kmeans.fit(X3)

##Implementing t-SNE to visualize dataset
#tsne = TSNE(n_components = 2, perplexity =30 ,n_iter=5000)
#x_tsne = tsne.fit_transform(X3)

#free = list(dataset['free sulfur dioxide'])
#total = list(dataset['total sulfur dioxide'])
#sulphates = list(dataset['sulphates'])

#data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
#                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
#                                text=[f'Free: {a}; Total: {b};  Sulphates:{c}' for a,b,c in list(zip(free,total,sulphates))],
#                                hoverinfo='text')]

#layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
#                    xaxis = dict(title='First Dimension'),
#                    yaxis = dict(title='Second Dimension'))
#fig = go.Figure(data=data, layout=layout)
#offline.plot(fig,filename='t-SNE3.html')

#Final Analysis to find labels for the Drink dataset
# Finding the number of clusters (K) - Elbow Plot Method
#inertia = []
#for i in range(1,11):
#    kmeans = KMeans(n_clusters = i, random_state = 100)
#    kmeans.fit(X4)
#    inertia.append(kmeans.inertia_)

#plt.plot(range(1, 11), inertia)
#plt.title('The Elbow Plot')
#plt.xlabel('Number of clusters')
#plt.ylabel('Inertia')
#plt.show()

# Running KMeans to generate labels
#kmeans = KMeans(n_clusters = 3)
#kmeans.fit(X4)

##Implementing t-SNE to visualize dataset
#tsne = TSNE(n_components = 2, perplexity =30 ,n_iter=5000)
#x_tsne = tsne.fit_transform(X4)

#free = list(dataset['free sulfur dioxide'])
#total = list(dataset['total sulfur dioxide'])
#sugar = list(dataset['residual sugar'])
#alcohol = list(dataset['alcohol'])
#fixed = list(dataset['fixed acidity'])
#volatile = list(dataset['volatile acidity'])

#data = [go.Scatter(x=x_tsne[:,0], y=x_tsne[:,1], mode='markers',
#                    marker = dict(color=kmeans.labels_, colorscale='Rainbow', opacity=0.5),
#                                text=[f'Free: {a}; Total: {b};  Sugar:{c}; Alcohol:{d}; FixedAcidity:{e}; VolatileAcidity:{f}' for a,b,c,d,e,f in list(zip(free,total,sugar,alcohol,fixed,volatile))],
#                                hoverinfo='text')]

#layout = go.Layout(title = 't-SNE Dimensionality Reduction', width = 700, height = 700,
#                    xaxis = dict(title='First Dimension'),
#                    yaxis = dict(title='Second Dimension'))
#fig = go.Figure(data=data, layout=layout)
#offline.plot(fig,filename='t-SNE4.html')
