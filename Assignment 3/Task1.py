import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error,r2_score
from sklearn import tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import sklearn.model_selection
import numpy as np

class Task1:

    # please feel free to create new python files, adding functions and attributes to do training, validation, testing
    def __init__(self):
        print("================Task 1================")
        '''self.df1
        self.df2
        self.x_train
        self.y_train
        self.x_test
        self.y_test
        self.lm'''
        return
    #Loading the train and test datasets as filename1 and filename2 in dataframe1 and dataframe2 respectively
    def load_data(self,filename1,filename2):
        self.df1 = pd.read_csv(filename1, header=None, delimiter='\t', na_values="n/a")
        self.df2 = pd.read_csv(filename2, header=None, delimiter='\t', na_values="n/a")
        #print(self.df1)
       # print(self.df2)

#Preprocess the datasets by performing one hot encoding and using pandas get_dummies which converts categorical
 #variable into dummy/indicator variables
    def preprocess(self, classcolumnname):
        one_hot_encoded_training_predictors = pd.get_dummies(self.df1)
        self.x_train = one_hot_encoded_training_predictors.drop(classcolumnname, axis=1)  # (585,56)
        self.y_train = one_hot_encoded_training_predictors[classcolumnname]  # (585,)
        self.df2['0_GP'] = int('0')#Missing columns after encoding are added with values initialized to 0 in test dataset
        self.df2['9_health'] = int('0')
        self.df2['15_school'] = int('0')
        self.df2['15_school family paid'] = int('0')
        self.df2['15_school  paid'] = int('0')
        one_hot_encoded_testing_predictors = pd.get_dummies(self.df2)
        self.x_test = one_hot_encoded_testing_predictors.drop(classcolumnname, axis=1)  # (64,56)
        self.y_test = one_hot_encoded_testing_predictors[classcolumnname]  # (64,)

#Train the model using linear regression,perform cross validation and obtain mean
    def trainlinearregression(self):
        self.lm=LinearRegression()
        self.lm.fit(self.x_train,self.y_train)
        lm_scores = sklearn.model_selection.cross_val_score(self.lm, self.x_train, self.y_train, cv=10).mean
        return lm_scores
#Train the model using decision tree regressor,perform cross validation
    def traindecisiontreeregressor(self):
        regressor = DecisionTreeRegressor(random_state=0, criterion="mae")
        self.dt = regressor.fit(self.x_train, self.y_train)
        dt_scores = sklearn.model_selection.cross_val_score(self.dt, self.x_train, self.y_train, cv=10)
        return dt_scores

#Calling the functions and evaluating learned model on testing data for linear regression and obtaining performance metric like coefficient,mean squared error and variance score
    def model_1_run(self):
        print("Model 1:")
        filename1='assign3_students_train.txt'
        filename2='assign3_students_test.txt'
        classcolumnname = 27
        self.load_data(filename1,filename2)
        self.preprocess(classcolumnname)
        lm_scores = self.trainlinearregression()
        # evaluate learned model on testing data
        print('print the performance of learned model 1 on testing data here')
        prediction = self.lm.predict(self.x_test)
        print(prediction)
        print('Coefficients: \n', self.lm.coef_)
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(self.y_test, prediction))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(self.y_test, prediction))
        return

#Calling the fucnctions and evaluating learned model on testing data for decision tree regressor and obtaining performance metrics
    def model_2_run(self):
        print("--------------------\nModel 2:")
        filename1 = 'assign3_students_train.txt'
        filename2 = 'assign3_students_test.txt'
        classcolumnname = 27
        self.load_data(filename1,filename2)
        self.preprocess(classcolumnname)

        dt_scores = self.traindecisiontreeregressor()
        # evaluate learned model on testing data
        print('print the performance of learned model 2 on testing data here')
        pred = self.dt.predict(self.x_test)
        print(pred)
        print("mean cross validation score: {}".format(np.mean(dt_scores)))
        print("score without cv: {}".format(self.dt.score(self.x_train, self.y_train)))
        # The mean squared error
        print("Mean squared error: %.2f"
              % mean_squared_error(self.y_test, pred))
        # Explained variance score: 1 is perfect prediction
        print('Variance score: %.2f' % r2_score(self.y_test, pred))

        return

if __name__ == '__main__':
    task1 = Task1()
    task1.model_1_run()
    task1.model_2_run()