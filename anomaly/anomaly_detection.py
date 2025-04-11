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
