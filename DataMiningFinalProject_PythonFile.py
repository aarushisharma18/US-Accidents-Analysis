#!/usr/bin/env python
# coding: utf-8

# # **Data Mining Project**

# **U.S. Accidents Analysis**

# **Group 15:**
# 
# **1. Sanil Rodrigues**
# 
# **2. Aarushi Sharma**

# Import necessary libraries for the project.

# In[1]:


get_ipython().system(' pip install plotly')


# In[86]:


import pandas as pd
import numpy as np
import seaborn as sns
import plotly.express as px
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, accuracy_score, classification_report
from sklearn import tree
import math
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.naive_bayes import GaussianNB
from sklearn import svm
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score


# **Connecting Google drive to google colab for importing dataset.**

# In[3]:


# from google.colab import drive
# drive.mount('/content/drive')
# %cd '/content/drive/MyDrive/Data'


# In[4]:


df_accidents = pd.read_csv("US_Accidents_Dec21_updated (1).csv")
df_accidents = df_accidents.reset_index(drop=True)


# In[5]:


df_accidents


# **Checking the percentage of null values in each column for original dataset.**

# In[6]:


df_accidents.isnull().sum() * 100 / len(df_accidents)


# **Dropping all the null values.**
# 
# 
# - Before dropping null values:
#       Original Dataset = 2.8 million records
# 
# - After dropping null values:
#       Dataset = 0.9 million records
# 

# In[7]:


df_drop = df_accidents.dropna().reset_index(drop=True)
# df_drop


# **Dropping variables not required for Exploratory Data Analysis and/or Model Building.**

# In[8]:


df_drop = df_drop.drop(columns = ['Number','ID','End_Lat','End_Lng','Zipcode','Country','Airport_Code','Wind_Chill(F)','Weather_Timestamp','Precipitation(in)'])
# df_drop


# **Checking the null values after dropping irrelevant variables.**

# In[9]:


df_drop.isnull().sum() * 100 /len(df_drop)


# In[10]:


df_viz = df_drop


# In[11]:


df_viz.insert(3,'Start_Date', pd.to_datetime(df_viz['Start_Time'], errors='coerce').dt.date)
df_viz['Start_Time'] = pd.to_datetime(df_viz['Start_Time'], errors='coerce').dt.time
df_viz.insert(5,'End_Date', pd.to_datetime(df_viz['End_Time'], errors='coerce').dt.date)
df_viz['End_Time'] = pd.to_datetime(df_viz['End_Time'], errors='coerce').dt.time


# In[12]:


df_viz.info()


# ## **DATA VISUALIZATION**

# **United States Map of Accidents according to Severity**

# In[13]:


sns.set(rc = {'figure.figsize':(20,15)})
sns.scatterplot(x ="Start_Lng", y = "Start_Lat", hue="Severity", data=df_viz)


# **City Level Analysis**

# In[14]:


city_df = pd.DataFrame(df_viz['City'].value_counts()).reset_index().rename(columns={'index':'City', 'City':'Cases'}).head(20)
fig = px.bar(city_df, y='Cases', x='City', text_auto='.2s', title="City Level analysis")
fig.show()


# **State Level Analysis**

# In[15]:


state_df = pd.DataFrame(df_viz['State'].value_counts()).reset_index().rename(columns={'index':'State', 'State':'Cases'}).head(20)
fig = px.bar(state_df, y='Cases', x='State', text_auto='.2s',
            title="State Level analysis")
fig.show()


# **United States Map of Accidents according to Time-Zones**

# In[16]:


sns.set(rc = {'figure.figsize':(20,15)})
sns.scatterplot(x ="Start_Lng", y = "Start_Lat", hue="Timezone", data=df_viz)


# **Street Level Analysis**

# In[17]:


street_df = pd.DataFrame(df_viz['Street'].value_counts()).reset_index().rename(columns={'index':'Street', 'Street':'Cases'}).head(20)
fig = px.bar(street_df, y='Cases', x='Street', text_auto='.2s', title="Street Level analysis")
fig.show()


# **Distibution of all levels of Severity**

# In[18]:


fig = px.pie(df_viz, values='Severity', names='Severity')
fig.show()


# NOTE: To view the Severity level for each distribution, hover over the pie chart.

# **Weather Analysis**

# In[19]:


df_weather = df_viz.groupby('Weather_Condition')['Severity'].value_counts().head(10)
weather_df = pd.DataFrame(df_weather)
weather_df = weather_df.rename(columns={'Severity':'Cases'})
# df.columns = ['new_col1', 'new_col2', 'new_col3', 'new_col4']
weather_df = weather_df.reset_index()


# In[20]:


# df_weather = pd.DataFrame(df.groupby('Severity')['Weather_Condition'].value_counts().head(10)).reset_index().rename(columns={'index':'Weather_Condition', 'Weather_Condition':'Cases','Severity':'severity'})
fig = px.bar(weather_df, y='Cases', x='Weather_Condition', text_auto='.2s', color="Severity", barmode="group", title="Weather analysis")
fig.show()


# NOTE: To view the number of cases of different levels of Severity, hover over the parts of stacked bar chart. 

# **Label Encoding of Categorical Variables for Model Building.**

# In[21]:


df_1 = df_viz.iloc[:,0:10]
# df_1


# In[22]:


from sklearn import preprocessing

label_encode = preprocessing.LabelEncoder()

df_2 =  df_viz[['Side', 'City','County','State','Timezone','Amenity','Bump','Crossing','Give_Way','Junction','No_Exit',
                      'Railway','Station','Stop','Traffic_Calming','Traffic_Signal','Turning_Loop','Sunrise_Sunset','Civil_Twilight',
                      'Nautical_Twilight','Astronomical_Twilight','Wind_Direction','Weather_Condition','Roundabout']].apply(label_encode.fit_transform)
# df_2                      


# In[23]:


df_encoded = pd.concat([df_1, df_2], axis = 1, join='inner')
df_encoded


# **Dropping columns not required for model building.**

# In[24]:


df_encoded = df_encoded.drop(columns=['Start_Time','End_Time','Start_Date','End_Date','Start_Lng','Start_Lat','Description','Street'])
# df_encoded.info()


# **Splitting the dataset into Input Dataset and Target Dataset.**

# In[25]:


X = df_encoded.drop(columns=['Severity']).sample(n=100000, random_state=1).reset_index(drop=True)
Y = df_encoded[['Severity']].sample(n=100000,random_state=1).reset_index(drop=True)


# **Standardizing the Input Dataset.**

# In[26]:


from sklearn.preprocessing import StandardScaler

feature_scale = StandardScaler()

X_stand = feature_scale.fit_transform(X)
X_stand = pd.DataFrame(X_stand, columns = X.columns).reset_index(drop=True)


# In[27]:


X_stand


# **Checking the correlation between the input variables to verify the redundant variables.**

# In[28]:


df_X_corr = X_stand.corr(method = 'pearson')
df_X_corr


# In[29]:


sns.set(rc = {'figure.figsize' : (20,20)})
sns.heatmap(df_X_corr, annot = True)


# Applying Principal Components Analysis

# In[30]:


pca_comp = PCA(n_components=20)
pca_final = pca_comp.fit_transform(X_stand)


# In[31]:


pca_score = pd.DataFrame(pca_final, columns = ['PCA1','PCA2','PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PCA11','PCA12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20'])
# df_target = pd.concat([pca_score, Y], axis=1, join='inner')
# df_target.columns=['PCA1', 'PCA2', 'PC3','PC4','PC5','PC6','PC7','PC8','PC9','PC10','PCA11','PCA12','PC13','PC14','PC15','PC16','PC17','PC18','PC19','PC20','Severity Target']
# df_target
pca_score


# In[32]:


explained_variance = pca_comp.explained_variance_
print('Explained Variance =', explained_variance)


# In[33]:


proportion_variance = pca_comp.explained_variance_ratio_
print('Proportion of Variance =', proportion_variance)


# In[34]:


cummulative_sum_exp = np.cumsum(proportion_variance)
print('Cummulative Proportion of Variance =', cummulative_sum_exp)


# **Dropping the redundant variables.**

# **Checking for variables having greater than 0.7 pearson correlation coefficient..**

# In[35]:


upper_tri = df_X_corr.where(np.triu(np.ones(df_X_corr.shape),k=1).astype(np.bool))
# upper_tri
df_corr_remove = [column for column in upper_tri.columns if any(upper_tri[column] > 0.70 )]
df_corr_remove = pd.DataFrame(df_corr_remove).rename(columns={0:'col_drop'})


# In[36]:


df_corr_remove


# In[37]:


X = X_stand.drop(columns = df_corr_remove['col_drop'])
X = X.drop(columns = ['Turning_Loop'])
X


# In[ ]:





# ## **MODEL BUILDING AND IMPLEMENTATION**

# **Splitting the Input and Target Data into Training and Test Dataset with 75% and 25% ratio, i.e., 75% - Training Data and 35% Test Data.**

# In[38]:


Y = Y.values
Y = Y.ravel()
Y


# In[39]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.25, random_state=4)


# In[40]:


X_train = X_train.reset_index(drop=True)
X_test = X_test.reset_index(drop=True)


# In[41]:


X_train


# **KNN-Classification**

# In[42]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

lis = [1,2,3,4,5,6,7,8,9,10]

acc_rmse = pd.DataFrame(columns=["K","Accuracy", "RMSE"])
for i in lis:
  k = i
  neigh = KNeighborsClassifier(n_neighbors = k).fit(X_train,y_train)
  pred_y = neigh.predict(X_test)
  acc_rmse = acc_rmse.append({'K': i, 'Accuracy': metrics.accuracy_score(y_test, pred_y), 'RMSE':metrics.mean_squared_error(y_test, pred_y)}, ignore_index=True)
  sns.lineplot(data=acc_rmse, x="K", y="Accuracy")


# In[43]:


acc_rmse


# Implementing KNN-Classification Model with n_neighbours = 10 yielding the best accuracy.

# In[44]:


neigh = KNeighborsClassifier(n_neighbors = 8).fit(X_train,y_train)

Pred_y = neigh.predict(X_test)

acc_knn = accuracy_score(y_test,Pred_y)
# acc_knn = mean_squared_error(y_test, pred_y)

Predicted_Y = pd.DataFrame(Pred_y).rename(columns ={0:'Predicted Y'})


# In[45]:


Predicted_Y


# In[46]:


# cm_knn = confusion_matrix(y_test, Pred_y, labels=Predicted_Y['Predicted Y'].values, sample_weight=None, normalize=None)


# In[47]:


import matplotlib.pyplot as plt


# In[48]:


# disp = ConfusionMatrixDisplay(confusion_matrix = cm_knn)
# disp.plot()
# plt.show()

fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    Pred_y,
    display_labels=["1", "2", "3", "4"],
    cmap='YlOrRd',
    ax=ax,
    colorbar=False).ax_.set(
                title='K-NN Classification Confusion Matrix')
plt.savefig("cm_knn", dpi=300)  # save the plot
plt.show()


# In[49]:


# # # plt.figure(figsize=(16,9))
# # # ax = sns.heatmap(cm_knn)
# # # plt.yticks(rotation=0)
# # # plt.show()

# sns.heatmap(cm_knn, square=True, annot=True, fmt='d', cbar=False, linewidths=.5)


# In[50]:


from sklearn.metrics import classification_report

print(classification_report(y_test, Pred_y))


# In[51]:


from sklearn.metrics import precision_recall_fscore_support as score

precision_knn,recall_knn,fscore_knn,support_knn=score(y_test,Pred_y,average='macro')

fscore_knn

Decision Tree Classification Model
# In[52]:


clf_en = tree.DecisionTreeClassifier(criterion='entropy', max_depth = 6).fit(X_train,y_train)
y_pred_tree = clf_en.predict(X_test)
acc_dt = metrics.accuracy_score(y_test, y_pred_tree)


# In[53]:


# cm_dt = confusion_matrix(y_test, y_pred_tree, labels=None, sample_weight=None, normalize=None)


# In[54]:


fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_tree,
    display_labels=["1", "2", "3", "4"],
    cmap='YlOrRd',
    ax=ax,
    colorbar=False).ax_.set(
                title='Decision Tree Confusion Matrix')
plt.savefig("cm_dt", dpi=300)  # save the plot
plt.show()


# In[55]:


print(classification_report(y_test, y_pred_tree))


# In[56]:


from sklearn.metrics import precision_recall_fscore_support as score

precision_dt,recall_dt,fscore_dt,support_dt=score(y_test,y_pred_tree,average='macro')

fscore_dt


# In[57]:


plt.figure(figsize=(25,10))
a = plot_tree(clf_en, 
              # feature_names = y_pred_tree.values_, 
              # class_names=y_train.columns, 
              filled=True, 
              rounded=True,
              fontsize=14)


# Logistic Regression Model

# In[58]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
 
y_pred_log = logmodel.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)


# In[58]:


from sklearn.linear_model import LogisticRegression

logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
 
y_pred_log = logmodel.predict(X_test)
acc_log = accuracy_score(y_test, y_pred_log)


# In[59]:


# cm_log = confusion_matrix(y_test, y_pred_log, labels=None, sample_weight=None, normalize=None)


# In[60]:


fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_log,
    display_labels=["1", "2", "3", "4"],
    cmap='YlOrRd',
    ax=ax,
    colorbar=False).ax_.set(
                title='Logistic Regression Confusion Matrix')
plt.savefig("cm_log", dpi=300)  # save the plot
plt.show()


# In[61]:


print(classification_report(y_test, y_pred_log))


# In[62]:


from sklearn.metrics import precision_recall_fscore_support as score

precision_log,recall_log,fscore_log,support_log=score(y_test,y_pred_log,average='macro')

fscore_log


# Naive Bayes Model

# In[63]:


from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
y_pred_nb  =  classifier.predict(X_test)
acc_naive = metrics.accuracy_score(y_test, y_pred_nb)


# In[64]:


# cm_naive = confusion_matrix(y_test, y_pred_nb, labels=None, sample_weight=None, normalize=None)


# In[65]:


fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_nb,
    display_labels=["1", "2", "3", "4"],
    cmap='YlOrRd',
    ax=ax,
    colorbar=False).ax_.set(
                title='Naive Bayes Confusion Matrix')
plt.savefig("cm_naive", dpi=300)  # save the plot
plt.show()


# In[66]:


print(classification_report(y_test, y_pred_nb))


# In[67]:


from sklearn.metrics import precision_recall_fscore_support as score

precision_nb,recall_nb,fscore_nb,support_nb=score(y_test,y_pred_nb,average='macro')

fscore_nb


# Support Vector Machine Model

# In[68]:


from sklearn import svm

#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
clf.fit(X_train, y_train)

#Predict the response for test dataset
y_pred_svm = clf.predict(X_test)

acc_svm = metrics.accuracy_score(y_test, y_pred_svm)


# In[69]:


# cm_svm = confusion_matrix(y_test, y_pred_svm, labels=None, sample_weight=None, normalize=None)


# In[70]:


fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    y_test,
    y_pred_svm,
    display_labels=["1", "2", "3", "4"],
    cmap='YlOrRd',
    ax=ax,
    colorbar=False).ax_.set(
                title='Support Vector Machine Confusion Matrix')
plt.savefig("cm_svm", dpi=300)  # save the plot
plt.show()


# In[71]:


print(classification_report(y_test, y_pred_svm))


# In[72]:


from sklearn.metrics import precision_recall_fscore_support as score

precision_svm,recall_svm,fscore_svm,support_svm=score(y_test,y_pred_svm,average='macro')

precision_svm


# Comparison of all implement models based on different parameters.

# Accuracy Comparison of all Models

# In[73]:


df_accuracy = pd.DataFrame({acc_knn, acc_dt, acc_log, acc_naive, acc_svm}).transpose().rename(columns = {0:'Acc_KNN', 1:'Acc_DT', 2:'Acc_Log', 3:'Acc_SVM', 4:'Acc_Naive'}).transpose()
df_accuracy = df_accuracy.reset_index().rename(columns={'index':'Model',0:'Accuracy'})
df_accuracy['Accuracy'] = df_accuracy['Accuracy'].apply(lambda x: x*100)
df_accuracy


# In[74]:


sns.barplot(data=df_accuracy, x='Model',y='Accuracy')


# Precision Comparison for all Models

# In[75]:


df_precision = pd.DataFrame({precision_knn, precision_dt, precision_log, precision_nb, precision_svm}).transpose().rename(columns = {0:'Precision KNN', 1:'Precision DT', 2:'Precision Log', 3:'Precision NB', 4:'Precision_SVM'}).transpose()
df_precision = df_precision.reset_index().rename(columns={'index':'Model',0:'Precision'})
# df_precision['Precision'] = df_accuracy['Precision'].apply(lambda x: x*100)
df_precision


# In[76]:


sns.barplot(data=df_precision, x='Model',y='Precision')


# F1 - Score Comparison of all Models

# In[80]:


df_f1score = pd.DataFrame({fscore_knn, fscore_dt, fscore_log, fscore_nb, fscore_svm}).transpose().rename(columns = {0:'F1 Score KNN', 1:'F1 Score SVM', 2:'F1 Score Log', 3:'F1 Score DT', 4:'F1 Score NB'}).transpose()
df_f1score = df_f1score.reset_index().rename(columns={'index':'Model',0:'F1 Score'})
# df_precision['Precision'] = df_accuracy['Precision'].apply(lambda x: x*100)
df_f1score


# In[81]:


sns.barplot(data=df_f1score, x='Model',y='F1 Score')


# In[ ]:




