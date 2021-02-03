#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#Custom Styling
sns.set_palette("Reds")
sns.set_style("darkgrid")
plt.style.use('seaborn')
mycmap = plt.get_cmap('gnuplot2')
plt.style.context('dark_background')

#Reading Dataset
df = pd.read_csv('Mall_Customers.csv', index_col=0)
print(df.head)
print(df.describe())
print(df.info())

#Exploratory Data Analysis
by_gender = df.groupby('Gender')
print(by_gender)
print(by_gender.size()/200*100)

# #Countplot
# sns.countplot(df['Gender'])
# plt.title('Distribution of  Gender')
# plt.show()
#
# #pairplot
# sns.pairplot(df, height=2.5)
# plt.title('Relationship of variables')
# plt.show()
#
# #Jointplot of income and age
# jp = sns.jointplot(df['Annual Income (k$)'], df['Age'], kind="kde", height=7, space=0, cmap=mycmap)
# plt.title('Income Vs. Age')
# plt.show()
#
# #Jointplot of income and spending score
# gi = sns.jointplot(df['Annual Income (k$)'], df['Spending Score (1-100)'], kind="kde", height=7, space=0, cmap=mycmap)
# plt.title('Income Vs. Spending Score')
# plt.show()
#
# #Jointplot of Age and spending score
# sa = sns.jointplot(df['Age'], df['Spending Score (1-100)'], kind="kde", height=7, space=0, cmap=mycmap)
# plt.title('Age Vs. Spending Score')
# plt.show()
#
# #Correlation Map
# sns.heatmap(df.corr(), annot=True, cmap='RdYlGn')
# plt.title('Correlation of the Variables')
#
# #Distribution of Annual income based on age and gender
# sns.relplot(x='Age', y='Annual Income (k$)', hue='Gender', sizes=(30, 300), height=8, alpha=.5, palette='muted', data=df)
# plt.title('Dist. Income based on Gender & Age')
# plt.show()
#
# #Encoding Gender
# df['Gender'] = df['Gender'].map({'Male':1, 'Female':0})
# print(df.head())
#
# #Boxplot of spending score and gender
# sns.boxplot(x='Gender', y='Spending Score (1-100)', hue='Gender', data=df, palette='Set2')
# plt.title('Spending Score Vs. Gender')
# plt.show()
#
# #VoilinPlot
# sns.catplot(x='Gender', y='Age', kind='violin', data=df, hue='Gender', split=False, palette='flare')
# plt.title('Gender Vs. Age')
# plt.show()
#
# #Age Distribution
# sns.displot(df['Age'], rug=True, color='b')
# plt.title(' Age Distribution')
# plt.show()
#
# #Spending Score Distribution
# sns.displot(df['Spending Score (1-100)'], rug=True, color='r')
# plt.title('Distribution of Spending Score')
# plt.show()
#
# #Annual Income Distribution
# sns.displot(df['Annual Income (k$)'], rug=True, color='g')
# plt.title('Distribution of Annual Income (k$)')
# plt.show()
#
# #Standardizing data
# scaler = StandardScaler()
# df_scaled = scaler.fit_transform(df)
#
# #statistics of scaled data
# print(pd.DataFrame(df_scaled).describe())
#
# #defining kmeans function with initializer as k-means++
# kmeans = KMeans(n_clusters=5, init='k-means++')
#
# #fitting the kmeans Algorithm on scaled data
# kmeans.fit(df_scaled)
#
# #inertia of the fitted data
# print(kmeans.inertia_)
# print(kmeans.cluster_centers_)
# print(kmeans.labels_)
# df['cluster label'] = kmeans.labels_
# print(df.head())
#
# #fitting multiple k-means and storing the values in an empty list
# ssd = []
# for cluster in range(1, 11):
#     km = KMeans(n_clusters=cluster, init='k-means++', max_iter=300, n_init=10, random_state=0)
#     km.fit(df_scaled)
#     ssd.append(km.inertia_)
# print(ssd)
#
# #Ploting Elbows graph
# plt.plot(range(1, 11), ssd)
# plt.title('Elbows Method', fontsize=30)
# plt.xlabel('No. of Clusters')
# plt.ylabel('Sum of Squared Distance')
# plt.show()
#
# #Clustering for Spending Score & Annual income
#
# X = df.loc[:,['Spending Score (1-100)', 'Annual Income (k$)']].values
#
# km = KMeans(n_clusters=5)
# km.fit(X)
# c = km.cluster_centers_
#
# color = np.array(['red', 'blue', 'green', 'yellow', 'purple'])
# labels = km.labels_
#
# fig, axs = plt.subplots(figsize=(8,8))
# fig.suptitle('KMeans Clustering (Spending Vs Income)', fontsize=22)
#
# axs.scatter(x=X[:,0], y=X[:,1], c=color[km.labels_], s=200)
# axs.scatter(x=c[:,0], y=c[:,1], marker='X', s=500, c='black')
#
# axs.set_xlabel('Spending Score (0-100)', fontsize=16)
# axs.set_ylabel('Income (k$)', fontsize=16)
#
# plt.tight_layout(rect=[0,0,1,.95])
# plt.show()
#
# #Clustering of Age and spending Score
# X = df.loc[:,['Spending Score (1-100)', 'Age']].values
#
# km = KMeans(n_clusters=5)
# km.fit(X)
# c = km.cluster_centers_
# color = np.array(['orange', 'blue', 'red', 'green', 'yellow'])
#
# labels = km.labels_
#
# fig, axs = plt.subplots(figsize=(8,8))
# fig.suptitle('Kmeans Clustering (Spending Vs Age)', fontsize=20)
#
# axs.scatter(x= X[:,0], y=X[:,1], c=color[km.labels_], s=200)
# axs.scatter(x=c[:,0], y=c[:,1], marker='X', s=500, c='black')
#
# axs.set_xlabel('Spending Score (1-100)', fontsize=16)
# axs.set_ylabel('Age', fontsize=16)
#
# plt.tight_layout(rect=[0,0,1,.95])
# plt.show()