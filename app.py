import streamlit as st 
from PIL import Image
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
st.set_option('deprecation.showfileUploaderEncoding', False)
# Load the pickled model
model = pickle.load(open('/content/kmeansclusterassignment.pkl','rb'))   
dataset= pd.read_csv('/content/clustering Dataset3.csv')
X = dataset.iloc[:,1:].values
# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'mean', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, [3,4,5,6,7,8]]) 
#Replacing missing data with the calculated mean value  
X[:, [3,4,5,6,7,8]]= imputer.transform(X[:, [3,4,5,6,7,8]])  
# Taking care of missing data
#handling missing data (Replacing missing data with the mean value)  
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values= np.NAN, strategy= 'most_frequent', fill_value=None, verbose=1, copy=True)
#Fitting imputer object to the independent variables x.   
imputer = imputer.fit(X[:, 1:3]) 
#Replacing missing data with the calculated mean value  
X[:, 1:3]= imputer.transform(X[:, 1:3])
# Encoding Categorical data:
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
def predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary):
  predict= model.predict(sc.transform([[CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary]]))
  print("cluster number", predict)
  if predict==[0]:
    prediction = "Customer is left"
  elif predict==[1]:
    prediction = "Customer is standard"
  elif predict==[2]:
    prediction = "Customer is Target"
  elif predict==[3]:
    prediction = "Customer is careful"
  else:
    prediction = "Custmor is sensible"
  print(prediction)
  return prediction

def main():
    
    html_temp = """
   <div class="" style="background-color:blue;" >
   <div class="clearfix">           
   <div class="col-md-12">
   <center><p style="font-size:40px;color:white;margin-top:10px;">Poornima Institute of Engineering & Technology</p></center> 
   <center><p style="font-size:30px;color:white;margin-top:10px;">Department of Computer Engineering</p></center> 
   <center><p style="font-size:25px;color:white;margin-top:10px;"Machine Learning Lab Experiment</p></center> 
   </div>
   </div>
   </div>
   """
    st.markdown(html_temp,unsafe_allow_html=True)
    st.header("Customer Segmenation on banking data ")
    CreditScore = st.text_input("CreditScore","")
    Geography = st.selectbox("Geography",("0","1"))
    Gender = st.selectbox("Gender",("0","1"))
    Age = st.text_input("Age","")
    Tenure = st.text_input("Tenure","")
    Balance = st.text_input("Balance","")
    HasCrCard = st.text_input("HasCrCard","")
    IsActiveMember = st.text_input("IsActiveMember","")
    EstimatedSalary = st.text_input("EstimatedSalary","")
    resul=""
    if st.button("Predict"):
      result=predict_note_authentication(CreditScore,Geography,Gender,Age,Tenure,Balance,HasCrCard,IsActiveMember,EstimatedSalary)
      st.success('Model has predicted {}'.format(result))
    if st.button("About"):
      st.text("Developed by Prabhjeet Singh")
      st.text("Student , Department of Computer Engineering")

if __name__=='__main__':
  main()