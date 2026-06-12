import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix,classification_report
import pickle
import warnings
warnings.filterwarnings("ignore")
class Nbex:
    def __init__(self,data):

        self.df = pd.read_csv(data)
    def dataprocessing(self):
        print(self.df.isnull().sum())
        self.df=self.df.drop(['Id'],axis=1)
        print(self.df.head(1))
        print(self.df['Species'].value_counts())
        self.df['Species']=self.df['Species'].map({'Iris-setosa':0,'Iris-versicolor':1,'Iris-virginica':2})
        print(self.df['Species'].unique())

    def splitingdata(self):
        self.X=self.df.iloc[:,:-1]
        self.y=self.df.iloc[:,-1]

        self.X_train,self.X_test,self.y_train,self.y_test= train_test_split(self.X,self.y,test_size=0.2,random_state=3)

    def modeltraining(self):
        self.reg=GaussianNB()
        self.reg.fit(self.X_train,self.y_train)
        self.predictions=self.reg.predict(self.X_train)
        self.trainconfusionmatrix=confusion_matrix(self.y_train,self.predictions)
        print(self.trainconfusionmatrix)
        self.report = classification_report(self.y_train,self.predictions)
        print(self.report)

    def testingdata(self):
        self.predict=self.reg.predict(self.X_test)
        self.testingconfusionmatrix = confusion_matrix(self.y_test,self.predict)
        print(self.testingconfusionmatrix)
        self.testreport= classification_report(self.y_test,self.predict)
        print(self.testreport)

    def  userprediction(self):
        sl=7.8
        sw=6.7
        pl=5.3
        pw=5.2
        self.userpredict=self.reg.predict([[sl,sw,pl,pw]])[0]
        print(self.userpredict)

    def pickle_file(self):
        with open("navie_bayes.pkl","wb") as f:
            pickle.dump(self.reg,f)







obj=Nbex("Iris.csv")
obj.dataprocessing()
obj.splitingdata()
obj.modeltraining()
obj.testingdata()
obj.userprediction()
obj.pickle_file()
