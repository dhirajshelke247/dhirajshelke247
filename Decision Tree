#!/usr/bin/env python
# coding: utf-8

# # Assignment on Decision Tree Classifier:
# A dataset collected in a cosmetics shop showing details of customers and whether or not they
# responded to a special offer to buy a new lip-stick is shown in table below. Use this dataset to
# build a decision tree, with Buys as the target variable, to help in buying lip-sticks in the future.
# Find the root node of decision tree. According to the decision tree you have made from
# previous training data set, what is the decision for the test data: [Age < 21, Income = Low,
# Gender = Female, Marital Status = Married]?

# In[38]:


import numpy as np
import pandas as pd


# In[39]:


data=pd.read_csv("sales.csv") 
data


# In[40]:


data.describe()


# In[41]:


#data.groupby('Age').count()


# In[54]:


data['Buys'].value_counts()


# In[43]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder();

x=data.iloc[:,:-1] #-1 means don't take last column 
x=x.apply(le.fit_transform)

print("Age with encodd value :",list( zip(data.iloc[:,0], x.iloc[:,0])))
print("\nIncome with encoded value :",list( zip(data.iloc[:,1], x.iloc[:,1])))
print("\nGender with encoded value :",list( zip(data.iloc[:,2], x.iloc[:,2])))
print("\nmaritialStatus with encoded value :",list( zip(data.iloc[:,3], x.iloc[:,3])))


# In[44]:


y=data.iloc[:,-1]


# In[45]:


from sklearn.tree import DecisionTreeClassifier
classifier=DecisionTreeClassifier(criterion='entropy')
classifier.fit(x,y)


# In[46]:


test_x=np.array([1,1,0,0])
pred_y=classifier.predict([test_x])
print("Predicted class for input [Age < 21, Income = Low,Gender = Female, Marital Status = Married]\n", test_x," is ",pred_y[0])


# In[49]:


from sklearn.tree import export_graphviz
from IPython.display import Image
export_graphviz(classifier,out_file="data.dot",feature_names=x.columns,class_names=["No","Yes"])

get_ipython().system('dot -Tpng data.dot -o tree.png')
Image("tree.png")


# In[55]:


from sklearn.externals.six import StringIO
import pydotplus as pdd
from IPython.display import Image
from sklearn.tree import export_graphviz

dot_dat=export_graphviz(classifier,out_file=None,feature_names=x.columns,class_names=["No","Yes"])
graph=pdd.graph_from_dot_data(dot_dat)
graph.write_png("tree.png")
Image(graph.create_jpg())


# In[56]:


#method 3
import pydotplus as pdd
from IPython.display import Image
dot_data = export_graphviz(classifier, out_file=None,feature_names=x.columns,class_names=['no', 'yes'], filled = True,special_characters=True)

graph = pdd.graph_from_dot_data(dot_data)  

Image(graph.create_png())
graph.write_png("dtree.png")
Image(graph.create_png())


# In[62]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
train,test=train_test_split(data.apply(le.fit_transform),test_size=0.14,random_state=0)
train_x=train.iloc[:,:-1]
train_y=train.iloc[:,-1]
test_x=test.iloc[:,:-1]
test_y=test.iloc[:,-1]
clf=DecisionTreeClassifier(criterion='entropy')
clf.fit(train_x,train_y)
pred_y=clf.predict(test_x)
accuracy=accuracy_score(test_y,pred_y)
accuracy*100


# In[59]:


test


# In[61]:


import seaborn as sns
corr=data.apply(le.fit_transform).corr();
sns.heatmap(corr,annot=True)


# In[ ]:




