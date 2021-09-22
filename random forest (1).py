#!/usr/bin/env python
# coding: utf-8

# In[1]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import plot_roc_curve, plot_precision_recall_curve, roc_curve, classification_report
import pandas as pd
from sklearn.metrics import accuracy_score
import numpy as np
df = pd.read_csv("FinalData.csv")
df.head()


# In[2]:


newdf=df.drop(['Stages','History','Patient','ControlledDiet','TakeMedication'], axis = 1)
newdf


# In[3]:


from sklearn import preprocessing
le = preprocessing.LabelEncoder()


# In[4]:


df2=(newdf.apply(le.fit_transform))
print(df2)


# In[5]:


result = pd.concat([df, df2], axis=1, join="inner")
result


# In[6]:


N=13
finalresult = result.iloc[: , N:]
finalresult


# In[7]:


target = finalresult.Stages
inputs = finalresult.drop('Stages',axis='columns')


# In[8]:


x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[9]:


len(x_train), len(y_train)


# In[10]:


len(x_test), len(y_test)


# # RandomForestClassifier

# In[11]:


from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier(n_estimators=50)
model.fit(x_train,y_train)


# In[12]:


model.score(x_test, y_test)


# In[13]:


y_pred = model.predict(x_test)
print(y_pred)


# In[14]:


accuracy_score(y_test, y_pred)*100


# In[15]:


print(model.predict([[1,0,1,0,0,0,1,1,3]]))


# In[16]:


R = int(input("Enter the number of rows:"))
C = int(input("Enter the number of columns:"))
  
  
print("Enter the entries in a single line (separated by space): ")
  
# User input of entries in a 
# single line separated by space
entries = list(map(int, input().split()))
  
# For printing the matrix
matrix = np.array(entries).reshape(R, C)
print(matrix)


# In[17]:


print(model.predict(matrix))


# # ConfusionMatrix_RandomForest

# In[18]:


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm


# In[19]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import seaborn as sn
plt.figure(figsize=(7,5))
sn.heatmap(cm, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # ClassificationReport_RandomForest

# In[20]:


print(classification_report(y_test, y_pred))


# # RocAucScore_RandomForest

# In[21]:


from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score


# In[22]:


y_score1 = roc_auc_score(y_test, model.predict_proba(x_test), multi_class='ovr')
y_score1


# # AdaBoostClassifier

# In[23]:


from sklearn.ensemble import AdaBoostClassifier
import warnings
warnings.filterwarnings('ignore')


# In[24]:


x_train, x_test, y_train, y_test = train_test_split(inputs,target,test_size=0.2)


# In[25]:


AdaModel = AdaBoostClassifier(n_estimators=100, learning_rate=1)


# In[26]:


model2 = AdaModel.fit(x_train,y_train)


# In[27]:


y_pred2 = model2.predict(x_test)
y_pred2


# In[28]:


accuracy_score(y_test, y_pred2)*100


# In[29]:


print(model2.predict([[1,0,1,0,0,0,1,1,3]]))


# # ConfusionMatrix_AdaBoost

# In[30]:


cm2 = confusion_matrix(y_test,y_pred2)
cm2


# In[31]:


plt.figure(figsize=(7,5))
sn.heatmap(cm2, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # ClassificationReport_AdaBoost

# In[32]:


print(classification_report(y_test, y_pred2))


# # RocAucScore_AdaBoost

# In[33]:


y_score2 = roc_auc_score(y_test, model2.predict_proba(x_test), multi_class='ovr')
y_score2


# # NaiveBayesClassifier

# In[34]:


from sklearn.naive_bayes import MultinomialNB


# In[35]:


model3 = MultinomialNB()
model3.fit(x_train, y_train)


# In[36]:


model3.score(x_test, y_test)


# In[37]:


y_pred3 = model3.predict(x_test)
print(y_pred3)


# In[38]:


print(model3.predict([[1,0,1,0,0,0,1,1,3]]))


# # ConfusionMatrix_NaiveBayes

# In[39]:


cm3 = confusion_matrix(y_test,y_pred3)
cm3


# In[40]:


plt.figure(figsize=(7,5))
sn.heatmap(cm3, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # ClassificationReport_NaiveBayes

# In[41]:


print(classification_report(y_test, y_pred3))


# # RocAucScore_NaiveBayes

# In[42]:


y_score3 = roc_auc_score(y_test, model3.predict_proba(x_test), multi_class='ovr')
y_score3


# # SupportVectorMachine(SVM)

# In[43]:


from sklearn.svm import SVC
model4 = SVC(probability=True)
model4.fit(x_train,y_train)


# In[44]:


model4.score(x_test, y_test)


# In[45]:


y_pred4 = model4.predict(x_test)
print(y_pred4)


# In[46]:


print(model4.predict([[1,0,1,0,0,0,1,1,3]]))


# # ConfusionMatrix_SVM

# In[47]:


cm4 = confusion_matrix(y_test,y_pred4)
cm4


# In[48]:


plt.figure(figsize=(7,5))
sn.heatmap(cm4, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # ClassificationReport_SVM

# In[49]:


print(classification_report(y_test, y_pred4))


# # RocAucScore_SVM

# In[50]:


y_score4 = roc_auc_score(y_test, model4.predict_proba(x_test), multi_class='ovr')
y_score4


# # DecisionTreeClassifier

# In[51]:


from sklearn.tree import DecisionTreeClassifier
from sklearn import tree


# In[52]:


model5 = DecisionTreeClassifier(max_leaf_nodes=10, random_state=0)
model5.fit(x_train,y_train)


# In[53]:


y_pred5 = model5.predict(x_test)
print(y_pred5)


# In[54]:


text_representation = tree.export_text(model5)
print(text_representation)


# In[55]:


with open("decistion_tree.log", "w") as fout:
    fout.write(text_representation)


# In[56]:


tree.plot_tree(model5);


# In[57]:


print(model5.predict([[1,0,1,0,0,0,1,1,3]]))


# In[58]:


model5.score(x_test, y_test)*100


# # ConfusionMatrix_DecisionTreeClassifier

# In[59]:


cm5 = confusion_matrix(y_test,y_pred5)
cm5


# In[60]:


plt.figure(figsize=(7,5))
sn.heatmap(cm5, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # ClassificationReport_DecisionTree

# In[61]:


print(classification_report(y_test, y_pred5))


# # RocAucScore_DecisionTree

# In[62]:


y_score5 = roc_auc_score(y_test, model5.predict_proba(x_test), multi_class='ovr')
y_score5


# # LogisticRegression

# In[63]:


from sklearn.linear_model import LogisticRegression
model6 = LogisticRegression()
model6.fit(x_train,y_train)


# In[64]:


model6.score(x_test, y_test)


# In[65]:


y_pred6 = model6.predict(x_test)
print(y_pred6)


# In[66]:


print(model6.predict([[1,0,1,0,0,0,1,1,3]]))


# In[67]:


print(model6.predict(matrix))


# # ConfusionMatrix_LogisticRegression

# In[68]:


from sklearn.metrics import confusion_matrix
cm6 = confusion_matrix(y_test,y_pred6)
cm6


# In[69]:


plt.figure(figsize=(7,5))
sn.heatmap(cm6, annot=True)
plt.xlabel('Predicted')
plt.ylabel('Truth')


# # ClassificationReport_LogisticRegression

# In[70]:


print(classification_report(y_test, y_pred6))


# # RocAucScore_LogisticRegression

# In[71]:


y_score6 = roc_auc_score(y_test, model6.predict_proba(x_test), multi_class='ovr')
y_score6


# # RandomForestClassifier_Medication

# In[72]:


mp=model.predict(matrix)
stage1 = 'HYPERTENSION (Stage-1)'
stage2 = 'HYPERTENSION (Stage-2)'
stage3 = 'HYPERTENSIVE CRISIS'
stage11 = 'HYPERTENSION (Stage-1).'
stage22 = 'HYPERTENSION (Stage-2).'
stage33 = 'HYPERTENSIVE CRISIS.'
norm = 'NORMAL'
if mp == stage1:
    print ("Medicine: Norvac")
elif mp == stage11:
    print ("medicine: Sofvasc")
elif mp == stage2:
    print ("medicine: Losarten")
elif mp == stage22:
    print ("medicine: Lipiget")
elif mp == stage3:
    print ("medicine: Natrilix")
elif mp == stage33:
    print ("medicine: Benefol")
 
else:
 
    print ("No medicine needed")


# # AdaBoost_Medication

# In[73]:


mp2=model2.predict(matrix)
stage1 = 'HYPERTENSION (Stage-1)'
stage2 = 'HYPERTENSION (Stage-2)'
stage3 = 'HYPERTENSIVE CRISIS'
stage11 = 'HYPERTENSION (Stage-1).'
stage22 = 'HYPERTENSION (Stage-2).'
stage33 = 'HYPERTENSIVE CRISIS.'
norm = 'NORMAL'
if mp2 == stage1:
    print ("Medicine: Ascard")
elif mp2 == stage11:
    print ("medicine: Sofvasc")
elif mp2 == stage2:
    print ("medicine: Losarten")
elif mp2 == stage22:
    print ("medicine: Lipiget")
elif mp2 == stage3:
    print ("medicine: Natrilix")
elif mp2 == stage33:
    print ("medicine: Benefol")
 
else:
 
    print ("No medicine needed")


# # NaiveBayes_Medication

# In[74]:


mp3=model3.predict(matrix)
stage1 = 'HYPERTENSION (Stage-1)'
stage2 = 'HYPERTENSION (Stage-2)'
stage3 = 'HYPERTENSIVE CRISIS'
stage11 = 'HYPERTENSION (Stage-1).'
stage22 = 'HYPERTENSION (Stage-2).'
stage33 = 'HYPERTENSIVE CRISIS.'
norm = 'NORMAL'
if mp3 == stage1:
    print ("Medicine: Norvac")
elif mp3 == stage11:
    print ("medicine: Byscard")
elif mp3 == stage2:
    print ("medicine: Losarten")
elif mp3 == stage22:
    print ("medicine: Lipiget")
elif mp3 == stage3:
    print ("medicine: Natrilix")
elif mp3 == stage33:
    print ("medicine: Benefol")
 
else:
 
    print ("No medicine needed")


# # SVM_Medication

# In[75]:


mp4=model4.predict(matrix)
stage1 = 'HYPERTENSION (Stage-1)'
stage2 = 'HYPERTENSION (Stage-2)'
stage3 = 'HYPERTENSIVE CRISIS'
stage11 = 'HYPERTENSION (Stage-1).'
stage22 = 'HYPERTENSION (Stage-2).'
stage33 = 'HYPERTENSIVE CRISIS.'
norm = 'NORMAL'
if mp4 == stage1:
    print ("Medicine: Norvac")
elif mp4 == stage11:
    print ("medicine: Sofvasc")
elif mp4 == stage2:
    print ("medicine: Covam ")
elif mp4 == stage22:
    print ("medicine: Lipiget")
elif mp4 == stage3:
    print ("medicine: Natrilix")
elif mp4 == stage33:
    print ("medicine: Benefol")
 
else:
 
    print ("No medicine needed")


# # DecisionTree_Medication

# In[76]:


mp5=model5.predict(matrix)
stage1 = 'HYPERTENSION (Stage-1)'
stage2 = 'HYPERTENSION (Stage-2)'
stage3 = 'HYPERTENSIVE CRISIS'
stage11 = 'HYPERTENSION (Stage-1).'
stage22 = 'HYPERTENSION (Stage-2).'
stage33 = 'HYPERTENSIVE CRISIS.'
norm = 'NORMAL'
if mp5 == stage1:
    print ("Medicine: Norvac")
elif mp5 == stage11:
    print ("medicine: Sofvasc")
elif mp5 == stage2:
    print ("medicine: Losarten")
elif mp5 == stage22:
    print ("medicine: Extro")
elif mp5 == stage3:
    print ("medicine: Natrilix")
elif mp5 == stage33:
    print ("medicine: Benefol")
 
else:
 
    print ("No medicine needed")


# # LogisticRegression_Medication

# In[77]:


mp6=model6.predict(matrix)
stage1 = 'HYPERTENSION (Stage-1)'
stage2 = 'HYPERTENSION (Stage-2)'
stage3 = 'HYPERTENSIVE CRISIS'
stage11 = 'HYPERTENSION (Stage-1).'
stage22 = 'HYPERTENSION (Stage-2).'
stage33 = 'HYPERTENSIVE CRISIS.'
norm = 'NORMAL'
if mp6 == stage1:
    print ("Medicine: Norvac")
elif mp6 == stage11:
    print ("medicine: Sofvasc")
elif mp6 == stage2:
    print ("medicine: Losarten")
elif mp6 == stage22:
    print ("medicine: Lipiget")
elif mp6 == stage3:
    print ("medicine: Natrilix")
elif mp6 == stage33:
    print ("medicine: Diabold")
 
else:
 
    print ("No medicine needed")

