
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score ,confusion_matrix
import seaborn as sns

dataset = pd.read_csv('creditcard.csv')
print(dataset)


#_________________________________________ Data preparation for classification ________________________________________


X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=True)


# Normalize data: train data -> fit_transform, test data -> transform
scaler = StandardScaler()

# Fit the scaler on the training data and transform the training data
X_train_normalized = scaler.fit_transform(X_train)
X_test_normalized=scaler.transform(X_test)

# Convert X_train_normalized to a DataFrame with matching indices and columns
X_train_normalized_df = pd.DataFrame(X_train_normalized, columns=X_train.columns, index=X_train.index)

# Convert y_train to a Series with matching index
y_train_series = pd.Series(y_train, name='target', index=X_train.index)

# Convert y_test to a Series with matching index
y_test_series = pd.Series(y_test, name='target', index=X_test.index)

# Concatenate X_train_normalized_df and y_train_series into a single DataFrame
train_data = pd.concat([X_train_normalized_df, y_train_series], axis=1)


X_test_normalized_df = pd.DataFrame(X_test_normalized, columns=X_test.columns, index=X_test.index)

# Concatenate X_test_normalized_df and y_test_series into a single DataFrame
test_data = pd.concat([X_test_normalized_df, y_test_series], axis=1)


X_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]


"""
# Check if rows have null values
null_rows = test_data.isnull().any(axis=1)
print("test_data - number of null rows:", sum(null_rows))
"""


# Separate Class 0 and Class 1 samples in the training set
class_0_train = train_data[train_data.iloc[:, -1] == 0]
class_1_train = train_data[train_data.iloc[:, -1] == 1]

# Determine the number of samples in each class in the training set
class_0_count_train = class_0_train.shape[0]
class_1_count_train = class_1_train.shape[0]

# Calculate the desired number of samples for Class 1 in the training set
desired_count_train = class_0_count_train

# Copy Class 1 samples in the training set until the desired count is reached
copstrain = class_1_train.sample(n=desired_count_train - class_1_count_train, replace=True)

# Concatenate the original training set, Class 1 samples, and copied samples
balanced_train_data = pd.concat([train_data, copstrain])

# Separate X_train and y_train from the balanced training set
X_train_balanced = balanced_train_data.iloc[:, :-1]
y_train_balanced = balanced_train_data.iloc[:, -1]

train_data = pd.concat([X_train_balanced, y_train_balanced], axis=1)

"""
#check if rows have null
null_rows=X_train_balanced.isnull().any(axis=1)
print("X_train_balanced----number of null rows:",sum(null_rows))
"""

#check for imbalancement
class_counts_balanced = y_train_balanced.value_counts()
print("class_counts_balanced: ")
print(class_counts_balanced)




X1 = train_data.iloc[:, :-1]
Y1 = train_data.iloc[:, -1]


X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, Y1, test_size=0.2, shuffle=True)



#check if rows have null
null_rows=X1.isnull().any(axis=1)
print("number of null rows:",sum(null_rows))




#_________________________________________clustering with actual dataset________________________________________
print("\n\n_________________________________________clustering with actual dataset________________________________________")
XX = dataset.iloc[:, :-1]
YY = dataset.iloc[:, -1]


model = KMeans(n_clusters=2, n_init=10)
model.fit(X=XX)
y_predicted = model.predict(XX)

accuracy = (YY == y_predicted).sum() / YY.count()
print("KMeans Accuracy with actual dataset:", accuracy)



cm = confusion_matrix(YY, y_predicted)

if cm[0][0] < cm[1][0]:
    print("first cluster related to class 1 ")
    print("second cluster related to class 0 ")
else:
    print("first cluster related to class 0 ")
    print("second cluster related to class 1 ")





X1_array = XX.values

# Plot each cluster separately
unique_labels = np.unique(y_predicted)
for label in unique_labels:
    cluster_data = X1_array[y_predicted == label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {label+1}")
    cluster_center = np.mean(cluster_data, axis=0)
    print(label," : ")
    print(f"Cluster center: {cluster_center}")
    cluster_mean = np.mean(cluster_data)
    cluster_variance = np.var(cluster_data)
    print(f"Number of data points: {len(cluster_data)}")
    print(f"Cluster mean: {cluster_mean}")
    print(f"Cluster variance: {cluster_variance}")


print("\n")


accuracy = accuracy_score(YY, y_predicted)
precision = precision_score(YY, y_predicted)
recall = recall_score(YY, y_predicted)
f1 = f1_score(YY, y_predicted)



print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)



plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Cluster Analysis for actual dataset')
plt.legend()
plt.show()



#_________________________________________clustering with prepared dataset________________________________________

print("\n\n_________________________________________clustering with prepared dataset________________________________________")
model = KMeans(n_clusters=2, n_init=10)
model.fit(X=X1)
y_predicted = model.predict(X1)

#accuracy = (Y1 != y_predicted).sum() / Y1.count()
#print("KMeans Accuracy with prepared dataset:", accuracy)

accuracy = accuracy_score(Y1, y_predicted)
precision = precision_score(Y1, y_predicted)
recall = recall_score(Y1, y_predicted)
f1 = f1_score(Y1, y_predicted)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)







X1_array = X1.values

# Plot each cluster separately
unique_labels = np.unique(y_predicted)
for label in unique_labels:
    cluster_data = X1_array[y_predicted == label]
    plt.scatter(cluster_data[:, 0], cluster_data[:, 1], label=f"Cluster {label+1}")

plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Cluster Analysis for prepared dataset')
plt.legend()
plt.show()






#_________________________________________classification with prepared dataset ________________________________________

print("\n\n_________________________________________classification with prepared dataset ________________________________________")
reg = LogisticRegression()
reg.fit(X_train1, y_train1)
y_pred = reg.predict(X_test)

q=(y_pred == y_test).sum() / y_test.count()

print("reg= ",q)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Calculate the confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Create a heatmap for the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix for classification ')
plt.show()

