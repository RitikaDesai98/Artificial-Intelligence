import pandas as pd
import numpy as np
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.model_selection import train_test_split
from sklearn .model_selection import cross_val_score
from pprint import pprint
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn .model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline

class Task3:

    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        return

    # Loading the train and test datasets as filename1 and filename2 in dataframe1 and dataframe2 respectively
    def load_data(self, filename1, filename2):
        self.df1 = pd.read_csv(filename1, header=None, delimiter='\t', na_values="n/a")
        self.df2 = pd.read_csv(filename2, header=None, delimiter='\t', na_values="n/a")
        #print(self.df1)
       # print(self.df2)

    # Preprocess the datasets by performing one hot encoding and using pandas get_dummies which converts categorical
    # variable into dummy/indicator variables
    def preprocess(self, classcolumnname):
        x_train = self.df1.drop(classcolumnname, axis=1)
        x_test = self.df2.drop(classcolumnname, axis=1)
        y_train = self.df1[classcolumnname]
        y_test = self.df2[classcolumnname]
        self.x_train = pd.get_dummies(x_train)
        self.y_train = pd.get_dummies(y_train)
        self.x_test = pd.get_dummies(x_test)
        self.y_test = pd.get_dummies(y_test)
        self.x_test['0_GP']=int('0')
        self.x_test['9_health']=int('0')
        self.y_test['9_health']=int('0')
        self.y_test['school paid'] = int('0')
        self.y_test['school'] = int('0')

    # Train the model using decison tree classifier,perform cross validation with value 10 and obtain dtree_model
    def decisiontreeclasiffier(self):
        dtree_model = DecisionTreeClassifier(max_depth=2).fit(self.x_train, self.y_train)
        self.scores = cross_val_score(dtree_model, self.x_train, self.y_train, cv=10)
        return dtree_model

    # Train the model using KNNclassifier,perform cross validation with value 10 and obtain cv_scores
    def KNNclassifier(self):
        self.knn = KNeighborsClassifier(n_neighbors=7)
        cv_scores = cross_val_score(self.knn, self.x_train, self.y_train, cv=10)
        self.knn.fit(self.x_train, self.y_train)
        return cv_scores

    # evaluate learned model on testing data for decision tree classifier and printing performance metrics like accuracy ,confusion matrix
    def model_1_run(self):
        print("Model 1:")
        filename1 = 'assign3_students_train.txt'
        filename2 = 'assign3_students_test.txt'
        classcolumnname = 15
        # train the model 1 with your best parameters on training data
        self.load_data(filename1, filename2)
        self.preprocess(classcolumnname)
        self.decisiontreeclasiffier()
        dtree_model = self.decisiontreeclasiffier()
        dtree_predictions = dtree_model.predict(self.x_test)
        accuracy = dtree_model.score(self.x_test, self.y_test)
        cm = confusion_matrix(self.x_test.values.argmax(axis=1), dtree_predictions.argmax(axis=1))
        # evaluate learned model on testing data
        print('print the performance of learned model 1 on testing data here')
        print("mean:{:3f}(std:{:3f})".format(self.scores.mean(), self.scores.std()), end="\n\n")
        # print(dtree_predictions)
        print("The accuracy of matrix is:",accuracy)
        print("The confusion matrix  is:\n",cm)
        return

    # evaluate learned model on testing data for  KNN classifier and printing performance metrics like accuracy ,confusion matrix etc
    def model_2_run(self):
        print("--------------------\nModel 2:")
        filename1 = 'assign3_students_train.txt'
        filename2 = 'assign3_students_test.txt'
        classcolumnname = 15
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
        print("The accuracy of the matrix:",accuracy)
        print("The confusion matrix is:\n ",cm)
        return


if __name__=='__main__':
    task3 = Task3()
    task3.model_1_run()
    task3.model_2_run()

