# -*- coding: utf-8 -*-
"""PIET18CS106-Prabhjeet_Singh-ML_Lab(Set-1).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1a2c2ltUq1HNvvMTjjsznLfkJRFY2Tjve
"""

# # Data Preprocessing
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from google.colab import files
uploaded = files.upload()

dataset= pd.read_csv('clustering Dataset3.csv')
dataset

#Check Missing Data
dataset.isnull().sum()

# Extracting dependent and independent variables:
# Extracting independent variable:
X = dataset.iloc[:, 1:].values

print(X)

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, [3,4,5,6,7,8]]) 
#Replacing missing data with the calculated mean value  
X[:, [3,4,5,6,7,8]]= imputer.transform(X[:, [3,4,5,6,7,8]])

print(X)

# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'most_frequent', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:3]) 
#Replacing missing data with the calculated mean value  
X[:, 1:3]= imputer.transform(X[:, 1:3])

types = dataset.dtypes
print(types)

#Count total number of classes in Data
class_counts = dataset.groupby('Geography').size()
print(class_counts)

#Count total number of classes in Data
class_counts = dataset.groupby('Gender').size()
print(class_counts)

# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

X[:,1:3]

# standardizing the data
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
# statistics of scaled data
pd.DataFrame(X).describe()

print(X)

print(np.isnan(X))

print(np.isnan(X).sum())

# Using the elbow method to find the optimal number of clusters
from sklearn.cluster import KMeans
wcss = []
for i in range(1, 20):
    kmeans = KMeans(n_clusters = i, init = 'k-means++', random_state = 42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 20), wcss)
plt.title('The Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

# Fitting K-Means to the dataset
kmeans = KMeans(n_clusters = 5, init = 'k-means++', random_state = 42)
print(kmeans)
y_kmeans = kmeans.fit_predict(X)

frame = pd.DataFrame(X)
frame['cluster'] = y_kmeans
frame['cluster'].value_counts()

print("Within cluster sum of square when k=5", kmeans.inertia_)

print("center of Cluster are", kmeans.cluster_centers_ )

print("Number of iterations", kmeans.n_iter_)

# Visualising the clusters
plt.scatter(X[:,0], X[:,5], s = 100, c = 'black', label = 'Data Distribution')
plt.title('Customer Distribution before clustering')
plt.xlabel('Channel')
plt.ylabel('Region)')
plt.legend()
plt.show()

frame = pd.DataFrame(X)
frame['cluster'] = y_kmeans
frame['cluster'].value_counts()

CreditScore=  1#@param {type:"number"}
Geography = 1 #@param {type:"number"}
Gender = 0 #@param {type:"number"}
Age =  42.0#@param {type:"number"}
Tenure =  2.0#@param {type:"number"}
Balance =  0.00#@param {type:"number"}
HasCrCard = 1.0 #@param {type:"number"}
IsActiveMember = 1.0 #@param {type:"number"}
EstimatedSalary = 101348.88 #@param {type:"number"}
predict= kmeans.predict([[ CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary ]])
print(predict)
if predict==[0]:
  print("Customer is not left")
else:
  print("Customer is left" )

# Visualising the clusters
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(X[y_kmeans== 4, 0], X[y_kmeans == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 3], s = 300, c = 'yellow', label = 'Centroids')
plt.title('Clusters of customers')
plt.xlabel('Spending on CreditScore')
plt.ylabel('Spending on Age')
plt.legend()
plt.show()

import pickle 
  
# Save the trained model as a pickle string. 
saved_model = pickle.dumps(kmeans) 
  
# Load the pickled model 
Saved_Model = pickle.loads(saved_model) 
  
# Use the loaded pickled model to make predictions 
Saved_Model.predict(X)

import pickle 
print("[INFO] Saving model...")
# Save the trained model as a pickle string. 
saved_model=pickle.dump(kmeans,open('/content/kmeansclusterassignment.pkl', 'wb')) 
# Saving model to disk

# Load the pickled model 
model = pickle.load(open('/content/kmeansclusterassignment.pkl','rb'))  
# Use the loaded pickled model to make predictions 
model.predict(X)

#!pip install streamlit

#!pip install pyngrok

#!ngrok authtoken 1sO9O2v7CGlRWPKUgjrtmB7tWIa_6unaAnmQHqgRwQhgTz8Jg

# Commented out IPython magic to ensure Python compatibility.
# %%writefile app.py
# import streamlit as st 
# from PIL import Image
# import pickle
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# st.set_option('deprecation.showfileUploaderEncoding', False)
# # Load the pickled model
# model = pickle.load(open('/content/kmeansclusterassignment.pkl','rb'))   
# dataset= pd.read_csv('/content/clustering Dataset3.csv')
# X = dataset.iloc[:,1:].values
# # Taking care of missing data
# #handling missing data (Replacing missing data with the mean value)  
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
# #Fitting imputer object to the independent variables x.   
# imputer = imputer.fit(X[:, [3,4,5,6,7,8]]) 
# #Replacing missing data with the calculated mean value  
# X[:, [3,4,5,6,7,8]]= imputer.transform(X[:, [3,4,5,6,7,8]])  
# # Taking care of missing data
# #handling missing data (Replacing missing data with the mean value)  
# from sklearn.impute import SimpleImputer
# imputer = SimpleImputer(missing_values= np.NAN, strategy= 'most_frequent', fill_value=None, verbose=1, copy=True)
# #Fitting imputer object to the independent variables x.   
# imputer = imputer.fit(X[:, 1:3]) 
# #Replacing missing data with the calculated mean value  
# X[:, 1:3]= imputer.transform(X[:, 1:3])
# # Encoding Categorical data:
# # Encoding the Independent Variable
# from sklearn.preprocessing import LabelEncoder
# labelencoder_X = LabelEncoder()
# X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
# X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
# from sklearn.preprocessing import StandardScaler
# sc = StandardScaler()
# X = sc.fit_transform(X)
# def predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
#   predict= model.predict(sc.transform([[CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]]))
#   print("cluster number", predict)
#   if predict==[0]:
#     prediction = "Customer is left"
#   elif predict==[1]:
#     prediction = "Customer is standard"
#   elif predict==[2]:
#     prediction = "Customer is Target"
#   elif predict==[3]:
#     prediction = "Customer is careful"
#   else:
#     prediction = "Custmor is sensible"
#   print(prediction)
#   return prediction
# 
# def main():
#     
#     html_temp = """
#    <div class="" style="background-color:blue;" >
#    <div class="clearfix">           
#    <div class="col-md-12">
#    <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
#    <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
#    <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
#    </div>
#    </div>
#    </div>
#    """
#     st.markdown(html_temp,unsafe_allow_html=True)
#     st.header("Customer Segmenation on banking data ")
#     CreditScore = st.text_input("CreditScore","")
#     Geography = st.selectbox("Geography",("0","1"))
#     Gender = st.selectbox("Gender",("0","1"))
#     Age = st.text_input("Age","")
#     Tenure = st.text_input("Tenure","")
#     Balance = st.text_input("Balance","")
#     HasCrCard = st.text_input("HasCrCard","")
#     IsActiveMember = st.text_input("IsActiveMember","")
#     EstimatedSalary = st.text_input("EstimatedSalary","")
#     resul=""
#     if st.button("Predict"):
#       result=predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
#       st.success('Model has predicted {}'.format(result))
#     if st.button("About"):
#       st.text("Developed by Prabhjeet Singh")
#       st.text("Student , Department of Computer Engineering")
# 
# if __name__=='__main__':
#   main()

#!nohup streamlit run  app.py &

#from pyngrok import ngrok
#url=ngrok.connect(port='8050')
#url

#!streamlit run --server.port 80 app.py
