import pandas as pd
import numpy as  np
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn .model_selection import cross_val_score
import numpy as np
from pprint import pprint
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix

import numpy as np
class Task2:

    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 2================")
        '''self.x_train
        self.x_test
        self.y_train
        self.y_test
        self.df1
        self.df2
        self.scores
        #self.a = None'''
        return

    # Loading the train and test datasets as filename1 and filename2 in dataframe1 and dataframe2 respectively
    def load_data(self, filename1, filename2):
        self.df1 = pd.read_csv(filename1, header=None, delimiter='\t', na_values="n/a")
        self.df2 = pd.read_csv(filename2, header=None, delimiter='\t', na_values="n/a")#(64,28)-33
        #print(self.df1)
        #print(self.df2)

    # Preprocess the datasets by performing one hot encoding and using pandas get_dummies which converts categorical
    # variable into dummy/indicator variables
    def preprocess(self,classcolumnname):
        self.df2['0_GP'] = int('0') #Missing columns after encoding are added with values initialized to 0 in test dataset
        self.df2['9_health'] = int('0')
        self.df2['15_school'] = int('0')
        self.df2['15_school family paid'] = int('0')
        self.df2['15_school  paid'] = int('0')
        x_train = self.df1.drop(classcolumnname, axis=1)
        x_test = self.df2.drop(classcolumnname, axis=1)
        y_train = self.df1[classcolumnname]
        y_test = self.df2[classcolumnname]
        self.x_train = pd.get_dummies(x_train)
        self.y_train = pd.get_dummies(y_train)
        self.x_test = pd.get_dummies(x_test)
        self.y_test = pd.get_dummies(y_test)
        self.y_test['health'] = int('0')

    # Train the model using decison tree classifier,perform cross validation with value 10 and obtain dtree_model
    def decisiontreeclasiffier(self):
        dtree_model = DecisionTreeClassifier(max_depth=2).fit(self.x_train, self.y_train)
        self.scores = cross_val_score(dtree_model, self.x_train, self.y_train, cv=10)
        return dtree_model

    # Train the model using KNNclassifier,perform cross validation with value 10 and obtain cv_scores
    def  KNNclassifier(self):
        self.knn = KNeighborsClassifier(n_neighbors=7)
        cv_scores = cross_val_score(self.knn, self.x_train, self.y_train, cv=10)
        self.knn.fit(self.x_train, self.y_train)
        return cv_scores

    # evaluate learned model on testing data for decision tree classifier and printing performance metrics like accuracy ,confusion matrix etc.
    def model_1_run(self):
        print("Model 1:")
        filename1 = 'assign3_students_train.txt'
        filename2 = 'assign3_students_test.txt'
        classcolumnname = 8
        # train the model 1 with your best parameters on training data
        self.load_data(filename1, filename2)
        self.preprocess(classcolumnname)
        self.decisiontreeclasiffier()
        dtree_model = self.decisiontreeclasiffier()
        dtree_predictions = dtree_model.predict(self.x_test)
        accuracy = dtree_model.score(self.x_test, self.y_test)
        cm = confusion_matrix(self.x_test.values.argmax(axis=1), dtree_predictions.argmax(axis=1))
        print('print the performance of learned model 1 on testing data here')
        print("mean:{:3f}(std:{:3f})".format(self.scores.mean(), self. scores.std()), end="\n\n")
        #print(dtree_predictions)
        print("The accuracy is:",accuracy)
        print("the confusion matrix is:\n",cm)
        return

    # evaluate learned model on testing data for  KNN classifier and printing performance metrics like accuracy ,confusion matrix etc.
    def model_2_run(self):
        print("--------------------\nModel 2:")
        filename1 = 'assign3_students_train.txt'
        filename2 = 'assign3_students_test.txt'
        classcolumnname = 8
        # train the model 2 with your best parameters on training data
        self.load_data(filename1, filename2)
        self.preprocess(classcolumnname)
        cv_scores = self.KNNclassifier()
        # evaluate learned model on testing data
        print('print the performance of learned model 2 on testing data here')
        knn_predictions = self.knn.predict(self.x_test)
        cm = confusion_matrix(
            self.y_test.values.argmax(axis=1), knn_predictions.argmax(axis=1))
        accuracy = self.knn.score(self.x_test, self.y_test)
        print(cv_scores)
        print('cv_scores mean:{}'.format(np.mean(cv_scores)))
        print("the accuracy is:",accuracy)
        print("the confusion matrix is:\n",cm)
        # cm = confusion_matrix(result2, knn_predictions)
        return

if __name__=='__main__':
    task2 = Task2()
    task2.model_1_run()
    task2.model_2_run()