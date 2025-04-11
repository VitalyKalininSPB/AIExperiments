import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings('ignore')
sns.set(style="whitegrid", font_scale=1.5)
plt.rcParams['figure.figsize'] = [12, 8]

# https://www.kaggle.com/datasets/nphantawee/
sensor_df = pd.read_csv("sensor.csv")

print("The dataset has", sensor_df.shape[0], "rows and", sensor_df.shape[1], "columns")
sensor_df.drop(["sensor_15", "Unnamed: 0"], inplace=True, axis=1)

# check for percentage of missing values in all columns
print(len(sensor_df.isnull().sum().sort_values(ascending=False))/len(sensor_df)*100)

print((sensor_df.isnull().sum().sort_values(ascending=False)/len(sensor_df))*100)
sensor_df.drop("sensor_50", inplace=True, axis=1)

# convert time into index
sensor_df['index'] = pd.to_datetime(sensor_df['timestamp'])
sensor_df.index = sensor_df['index']

# drop index and timestamp columns
sensor_df.drop(["index", "timestamp"], inplace=True, axis=1)

print(sensor_df.head())
sensor_df.drop(['machine_status'], inplace=True, axis=1)

# Change missing values to mean for each column
sensor_df.fillna(sensor_df.mean(), inplace=True)
#print(sensor_df.isnull().sum())
#print(sensor_df.machine_status.value_counts())

# machine status - pie chart

stroke_labels = ["Normal", "Recovering", "Broken"]
#sizes = sensor_df.machine_status.value_counts()
#plt.pie(x=sizes, autopct="%1.3f%%", labels=stroke_labels)
#plt.show()

# Extract the readings from the BROKEN state of the pump
#broken = sensor_df[sensor_df['machine_status']=='BROKEN']
# Extract the names of the numerical columns
#sensor_df_2 = sensor_df.drop(['machine_status'], axis=1)
#names = sensor_df_2.columns
# Plot time series for each sensor with BROKEN state marked with X in red color
'''
for name in names:
    _ = plt.figure(figsize=(18,3))
    _ = plt.plot(broken[name], linestyle='none', marker='X', color='red', markersize=12)
    _ = plt.plot(sensor_df[name], color='blue')
    _ = plt.title(name)
    plt.show()
'''

from sklearn.preprocessing import StandardScaler
# dropping the target column from the dataframe
#sensor_df_2 = sensor_df.drop("machine_status", axis=1)
col_names = sensor_df.columns

#scaling
scaler = StandardScaler()
sensor_df_2_scaled = scaler.fit_transform(sensor_df)
sensor_df_2_scaled = pd.DataFrame(sensor_df_2_scaled, columns=col_names)

from sklearn.decomposition import PCA
pca = PCA()
pca.fit(sensor_df_2_scaled)

# Plot the principal components
features = range(pca.n_components_)
_ = plt.figure(figsize=(22, 5))
_ = plt.bar(features, pca.explained_variance_)
_ = plt.xlabel('PCA feature')
_ = plt.ylabel('Variance')
_ = plt.xticks(features)
_ = plt.title("Important Principal Components")
plt.show()

# Calculate PCA with 2 components
pca = PCA(n_components=2)
pComponents = pca.fit_transform(sensor_df_2_scaled)
principal_df = pd.DataFrame(data = pComponents, columns = ['pc1', 'pc2'])

# Autocorrelation
# Compute change in daily mean
pca1 = principal_df['pc1'].pct_change()
# Compute autocorrelation
autocorrelation = pca1.dropna().autocorr()
print('Autocorrelation is: ', autocorrelation)

# Compute change in daily mean
pca2 = principal_df['pc2'].pct_change()
# Compute autocorrelation
autocorrelation = pca2.dropna().autocorr()
print('Autocorrelation is: ', autocorrelation)

# Plot ACF
from statsmodels.graphics.tsaplots import plot_acf
plot_acf(pca1.dropna(), lags=20, alpha=0.05)
plt.show()

# Modeling
# import kmeans
from sklearn.cluster import KMeans
# initialize and fit kmeans
kmeans = KMeans(n_clusters=2, random_state=13)
kmeans.fit(principal_df.values)

# prediction
labels = kmeans.predict(principal_df.values)

# plotting the clusters
_ = plt.figure(figsize=(9,7))
_ = plt.scatter(principal_df['pc1'], principal_df['pc2'], c=labels)
_ = plt.xlabel('pc1')
_ = plt.ylabel('pc2')
_ = plt.title('K-means of clustering')
plt.show()

# Write a function that calculates distance between each point and
# the centroid of the closest cluster

def getDistanceByPoint(data, model):
    """ Function that calculates the distance between a point and centroid of a cluster,
            returns the distances in pandas series"""
    distance = []
    for i in range(0,len(data)):
        Xa = np.array(data.loc[i])
        Xb = model.cluster_centers_[model.labels_[i]-1]
        distance.append(np.linalg.norm(Xa-Xb))
    return pd.Series(distance, index=data.index)

# Assume that 13% of the entire data set are anomalies
outliers_fraction = 0.13

# get the distance between each point and its nearest centroid.
# the biggest distances are considered as anomaly
distance = getDistanceByPoint(principal_df, kmeans)

# number of observations that equate to the 13% of the entire data set
number_of_outliers = int(outliers_fraction*len(distance))

# Take the minimum of the largest 13% of the distances as the threshold
threshold = distance.nlargest(number_of_outliers).min()

# anomaly1 contains the anomaly result of the above method Cluster (0:normal, 1:anomaly)
principal_df['kmeans_anomaly'] = (distance >= threshold).astype(int)

sensor_df['kmeans_anomaly'] = pd.Series(principal_df['kmeans_anomaly'].values, index=sensor_df.index)
a = sensor_df[sensor_df['kmeans_anomaly'] == 1] #anomaly
_ = plt.figure(figsize=(18,6))
_ = plt.plot(sensor_df['sensor_00'], color='blue', label='Normal')
_ = plt.plot(a['sensor_00'], linestyle='none', marker='X', color='red', markersize=12, label='KMeans Anomaly')
#_ = plt.plot(dfBroken['sensor_00'], linestyle='none', marker='X', color='green', markersize=12, label='Broken')
_ = plt.xlabel('Date and Time')
_ = plt.ylabel('Sensor Reading')
_ = plt.title('Sensor_00 Anomalies')
_ = plt.legend(loc='best')
plt.show()
