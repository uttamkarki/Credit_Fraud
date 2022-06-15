# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# classification libraries
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

# supporting libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# read dataset using pandas
df = pd.read_csv('creditcard.csv')
print(df.head())
print(df.describe())
print(df.isnull().sum().max())  # checking the null values in the datasets
print(df.columns)

# finding the number of fraud vs non-frauds cases in the dataset

Non_fraud = round((df['Class'].value_counts()[0] / len(df)) * 100, 2)
fraud = round((df['Class'].value_counts()[1] / len(df)) * 100, 2)
number_fraud = df['Class'].value_counts()[1]
print(number_fraud, "fraud cases are in the dataset")
print(fraud, '% of transactions are fraud')
print(Non_fraud, '% of transactions are non-fraud')

# 0.17% (Fraud cases) vs 99.83% (Non Fraud) cases. This is highly imbalance data

# lets plot the histogram using seaborn to shoe fraud vs non-fraud cases
sns.countplot('Class', data=df, palette=['green', 'red'])
plt.title("Distribution of Non-Fraud (0) vs Fraud (1) cases")
plt.show()

# distribution shows a very skewed features

# As the dataset is highly imbalance, we should balance it by sampling and taking 50/50 ration of fraud and non-fraud
# cases. If we train the dataset with the original data, as majority of the output are non-fraud, we might overfit
# the model and we will not have strong evidence how the input features actually influence the fraud vs non-fraud cases

# Additionally, as the Time and Amount columns are not scaled, lets scale them like other input features

# we will use sklearn to scale Time and Amount feature
robust_scaler = RobustScaler()
df['Time_scaled'] = robust_scaler.fit_transform(df['Time'].values.reshape(-1, 1))
df['Amount_scaled'] = robust_scaler.fit_transform(df['Amount'].values.reshape(-1, 1))

# now we can just drop original Time and amount column
df.drop(['Time', 'Amount'], axis=1, inplace=True)
print(df.head())

# lets move the scaled amount and time as first and second column
time_scaled = df['Time_scaled']
amount_scaled = df['Amount_scaled']
df.drop(['Time_scaled', 'Amount_scaled'], axis=1, inplace=True)
df.insert(0, 'Amount_scaled', amount_scaled)
df.insert(1, 'Time_scaled', time_scaled)
print(df.head())

# Before creating a training and testing dataset to train the logistic regression model, first lets store the
# original dataset. we will need original dataset to test the accuracy of our model.

# assigning X and Y for independent and dependent variable
X = df.drop('Class', axis=1)
y = df['Class']

# As our original dataset is highly skewed, we will have equal number of fraud and non fraud cases in out training set.
# creating a dataset of equal number of fraud and nonfraud cases

# shuffle the data
df = df.sample(frac=1)

df_fraud = df.loc[df['Class'] == 1]
df_non_fraud = df.loc[df['Class'] == 0][:number_fraud]  # number fraud is total fraud cases in the dataset i.e. 492

# merging two fraud and non fraud data frame into one using pandas concat
df_new = pd.concat([df_fraud, df_non_fraud])
df_new = df_new.sample(frac=1, random_state=42)  #
print(df_new.head())

# now lets again see the number of fraud vs non fraud cases in new data frame
Non_fraud_new = round((df_new['Class'].value_counts()[0] / len(df_new)) * 100, 2)
fraud_new = round((df_new['Class'].value_counts()[1] / len(df_new)) * 100, 2)
print(fraud_new, '% of transactions are fraud in new dataset')
print(Non_fraud_new, '% of transactions are non-fraud in new dataset')
plt.clf()
sns.countplot('Class', data=df_new, palette=['green', 'red'])
plt.title("Distribution of Non-Fraud (0) vs Fraud (1) cases")
plt.show()

# now lets see the correlation between input features (V1,V2...V28) and fraud cases.
# we will use seaborn heat map to see the correlation of the newly created sample data
plt.clf()
coreltn = df_new.corr()
sns.heatmap(coreltn, cmap='coolwarm_r', annot_kws={'size': 30})
plt.title("Correlation using newly created sample data")
plt.show()

# plot shows the correlation of input feature on fradulent activity (red vs blue color)
# feature with negative correlation on y are (V10,V12,V14,V17)
# feature with positive correlation on y are (V2,V4,V11,V19)


# we have feature with high correlation with out dependent variable. Lets check if there are any outliers with box plot
# first lets start with feature that are positively correlated with fraudulent activity
plt.clf()
f, axes = plt.subplots(ncols=4, figsize=(25, 5))  # we have 4 feature, so n = 4
sns.boxplot(x="Class", y="V2", data=df_new, palette=['green','red'], ax=axes[0])
axes[0].set_title('Positive Correlation: V2 vs Fraud cases')

sns.boxplot(x="Class", y="V4", data=df_new, palette=['green','red'], ax=axes[1])
axes[1].set_title('Positive Correlation: V4 vs Fraud cases')


sns.boxplot(x="Class", y="V11", data=df_new, palette=['green','red'], ax=axes[2])
axes[2].set_title('Positive Correlation: V11 vs Fraud cases')


sns.boxplot(x="Class", y="V19", data=df_new, palette=['green','red'], ax=axes[3])
axes[3].set_title('Positive Correlation: V19 vs Fraud cases')
plt.show()

# lets plot box plot with feature that are negatively correlated with fraudulent activity
plt.clf()
f, axes = plt.subplots(ncols=4, figsize=(25, 5))  # we have 4 feature, so n = 4
sns.boxplot(x="Class", y="V10", data=df_new, palette=['green','red'], ax=axes[0])
axes[0].set_title('Negative Correlation: V10 vs Fraud cases')

sns.boxplot(x="Class", y="V12", data=df_new, palette=['green','red'], ax=axes[1])
axes[1].set_title('Negative Correlation: V12 vs Fraud cases')

sns.boxplot(x="Class", y="V14", data=df_new, palette=['green','red'], ax=axes[2])
axes[2].set_title('Negative Correlation: V14 vs Fraud cases')

sns.boxplot(x="Class", y="V17", data=df_new, palette=['green','red'], ax=axes[3])
axes[3].set_title('Negative Correlation: V17 vs Fraud cases')

plt.show()
# from the box plot, we can see outliers as many observation are outside the box plots.
# since they are highly correlated with out fraud cases, we should consider removing them because they will reduce our
# model accuracy if we incorporate them while building a model

# Outlier removal for highly correlated input features. We will consider an observation as outlier if they are below or
# above 1.5*inter quartile range. We can always change this 1.5 value and see how the model accuracy changes.
# there is always a tug of war while tuning this 1.5 value

# for V10 input feature
fraud_input = df_new['V10'].loc[df_new['Class'] == 1].values
Q_25th, Q_75th = np.percentile(fraud_input, 25), np.percentile(fraud_input, 75)
IQ_input = Q_75th - Q_25th
lb_input, ub_input = Q_25th - (IQ_input*1.5), Q_75th + (IQ_input*1.5)
outliers = [x for x in fraud_input if x < lb_input or x > ub_input]
df_new = df_new.drop(df_new[(df_new['V10'] > ub_input) | (df_new['V10'] < lb_input)].index)

# for V12 input feature
fraud_input = df_new['V12'].loc[df_new['Class'] == 1].values
Q_25th, Q_75th = np.percentile(fraud_input, 25), np.percentile(fraud_input, 75)
IQ_input = Q_75th - Q_25th
lb_input, ub_input = Q_25th - (IQ_input*1.5), Q_75th + (IQ_input*1.5)
outliers = [x for x in fraud_input if x < lb_input or x > ub_input]
df_new = df_new.drop(df_new[(df_new['V12'] > ub_input) | (df_new['V12'] < lb_input)].index)

# for V14 input feature
fraud_input = df_new['V14'].loc[df_new['Class'] == 1].values
Q_25th, Q_75th = np.percentile(fraud_input, 25), np.percentile(fraud_input, 75)
IQ_input = Q_75th - Q_25th
lb_input, ub_input = Q_25th - (IQ_input*1.5), Q_75th + (IQ_input*1.5)
outliers = [x for x in fraud_input if x < lb_input or x > ub_input]
df_new = df_new.drop(df_new[(df_new['V14'] > ub_input) | (df_new['V14'] < lb_input)].index)

# now we are close to run our prediction algorithm as we have completed our data pre processing.
# before that, lets quickly split our sample data into train and test set using sklearn train_test_split.

X_new = df_new.drop('Class',axis=1)
y_new = df_new['Class']
# we are using 80/20rule for train adn test split
X_train, X_test, y_train, y_test = train_test_split(X_new, y_new, test_size=0.2, random_state=42)

# just data structuring to feed into out classification models
X_train = X_train.values
X_test = X_test.values
y_train = y_train.values
y_test = y_test.values

# let's train logistic regression
logit = LogisticRegression()
logit.fit(X_train,y_train)
# storing predicted values to create confusing matrix later
prediction = logit.predict(X_test)

# now we have fitted a logistic regression model, let test the accuracy of the model over testing dataset
pred_accuracy = round(logit.score(X_test,y_test)*100,2)
print("The accuracy of your trained logistic regression model is " ,pred_accuracy,"%")

logit_cf = confusion_matrix(y_test,prediction)
print(logit_cf)

# lets create a confusing matrix using seaborn heat map
plt.clf()
sns.heatmap(logit_cf,annot=True, cmap='Blues')
plt.title("Confusion Matrix (Logistic regression)\n 1 = Fraud || 0 = Non-Fraud")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# Now lets test the logistic regression with other classifier

# Decision tree classifier
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train,y_train)
prediction = decision_tree.predict(X_test)
pred_accuracy = round(decision_tree.score(X_test,y_test)*100,2)
print("The accuracy of your trained decision tree classifier model is " ,pred_accuracy,"%")
decision_cf = confusion_matrix(y_test,prediction)
print(decision_cf)

# lets create a confusing matrix using seaborn heat map
plt.clf()
sns.heatmap(decision_cf,annot=True, cmap='Blues')
plt.title("Confusion Matrix (Decision Tree Classifier)\n 1 = Fraud || 0 = Non-Fraud")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()

# K Neighbor Classifier
Knearest = KNeighborsClassifier()
Knearest.fit(X_train,y_train)
prediction = Knearest.predict(X_test)
pred_accuracy = round(Knearest.score(X_test,y_test)*100,2)
print("The accuracy of your trained K neighbor classifier model is " ,pred_accuracy,"%")
kneighbor_cf = confusion_matrix(y_test,prediction)
print(kneighbor_cf)

# lets create a confusing matrix using seaborn heat map
plt.clf()
sns.heatmap(kneighbor_cf,annot=True, cmap='Blues')
plt.title("Confusion Matrix (K Neighbor Classifier)\n 1 = Fraud || 0 = Non-Fraud")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


exit()
