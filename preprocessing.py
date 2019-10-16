# -*- coding: utf-8 -*-
"""
Created on Wed Oct  2 20:23:51 2019

@author: mankadp
"""

## DaTA preprocessing
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.optimizers import Adam
from keras.layers.core import Dense
from keras.models import Model
from keras.layers.core import Activation
from keras.layers import Dropout


def mytfidf(data,tfidf,fit=False):
    if fit ==True:
        tfidf.fit((data))
    data = tfidf.transform(data)
    data = pd.DataFrame(data.toarray())
    return data, tfidf

def mycountVectoriser(data,cvect,fit=False):
    if fit:
        cvect.fit(data)
    data = cvect.transform(data)
    data = data.toarray()
    data = pd.DataFrame(data)
    return data, cvect

def imputing(data):
    try:
        data = data.fillna("unknown")
        data = data.replace('0','unknown')
    except:
        data = pd.DataFrame(data)
        data = data.fillna("unknown")
        data = data.replace('0','unknown')
    return data
    
## finds the distinct no of values
def distinct(col):
    unique=[]
    for i in col:
        if i not in unique:
            unique.append(i)
            
    print(unique)
    return len(unique)

## DaTA preprocessing
def data_prep(data,test):
    labels = data.iloc[:,-1]
    ####################### TRAINING DATA PROCESSING ###################
    
    ## Data Separation 
    data_num = pd.concat([data["Age"],data["Year of Record"],data["Body Height [cm]"],data["Size of City"]],axis=1)
    data_cat = imputing(pd.concat([data["Gender"],data["University Degree"],data["Profession"],data["Country"],data["Hair Color"]],axis=1))
    
    
    gen = imputing(data["Gender"])
    deg = imputing(data["University Degree"])
    prof = imputing(data["Profession"])
    coun = imputing(data["Country"])
    hair = imputing(data["Hair Color"])
    
    
    ################## TEST DATA PREPROCESSING ###################
    
    # Test Data separation 
    test_num = pd.concat([test["Age"],test["Year of Record"],data["Body Height [cm]"],test["Size of City"]],axis=1)
    test_cat = imputing(pd.concat([test["Gender"],test["University Degree"]],axis=1))
    
    test_gen = imputing(test["Gender"])
    test_deg = imputing(test["University Degree"])
    test_prof = imputing(test["Profession"])
    test_coun = imputing(test["Country"])
    test_hair = imputing(data["Hair Color"])
    
    ################ Fitting And Transforming ######################
    
    ## Imputing numeric data
    from sklearn.impute import SimpleImputer
    imp = SimpleImputer(strategy="mean")
    imp.fit(data_num)
    data_num_imputed = imp.transform(data_num)
    test_num_imputed = imp.transform(test_num)
    
    age = data_num_imputed[:,0].astype(str)
    height = data_num_imputed[:,2].astype(str)
    size = data_num_imputed[:,3].astype(str)
    year = data_num_imputed[:,1].astype(str)
    test_age = test_num_imputed[:,0].astype(str)
    test_height = test_num_imputed[:,2].astype(str)
    test_size = test_num_imputed[:,3].astype(str)
    test_year = test_num_imputed[:,1].astype(str)
    
    
    ##### TFIDF Vectorizer #####
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    
    prof, tfidf = mytfidf(prof, tfidf, fit=True)
    test_prof, tfidf = mytfidf(test_prof, tfidf)
    
    coun, tfidf = mytfidf(coun, tfidf, fit=True)
    test_coun, tfidf = mytfidf(test_coun, tfidf)
    
    gen, tfidf = mytfidf(gen, tfidf, fit=True)
    test_gen, tfidf = mytfidf(test_gen, tfidf)
    
    deg, tfidf = mytfidf(deg, tfidf, fit=True)
    test_deg, tfidf = mytfidf(test_deg, tfidf)
    
    hair, tfidf = mytfidf(hair, tfidf, fit=True)
    test_hair, tfidf = mytfidf(test_hair, tfidf)
    ###
    age, tfidf = mytfidf(age, tfidf, fit=True)
    test_age, tfidf = mytfidf(test_age, tfidf)
    
    height, tfidf = mytfidf(height, tfidf, fit=True)
    test_height, tfidf = mytfidf(test_height, tfidf)
    
    #size, tfidf = mytfidf(size, tfidf, fit=True)
    #test_size, tfidf = mytfidf(test_size, tfidf)
    
    year, tfidf = mytfidf(year, tfidf, fit=True)
    test_year, tfidf = mytfidf(test_year, tfidf)
    
    data_vect2 = pd.concat([prof,coun,gen,deg,hair,age,height,year],axis=1)
    test_vect2 = pd.concat([test_prof,test_coun,test_gen,test_deg,test_hair,test_age,test_height,test_year],axis=1)
    
    #NOT BEING USED
    ## using Ordinal Encoder
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    oe = OrdinalEncoder()
    data_oe = oe.fit_transform(data_cat)
    data_oe = pd.DataFrame(data_oe)
    test_oe = oe.transform(test_cat)
    test_oe = pd.DataFrame(test_oe)
    '''
    ## One hot encoder
    ohe = OneHotEncoder()
    data_ohe = ohe.fit_transform(data_cat).toarray()
    data_ohe = pd.DataFrame(data_ohe)
    test_ohe = ohe.transform(test_cat).toarray()
    test_ohe = pd.DataFrame(test_ohe)
    '''
    
    ## Computing Fscore and probability values
    from sklearn.feature_selection import f_regression
    f, pval = f_regression(pd.concat([pd.DataFrame(data_num_imputed),data_oe],axis=1),labels)
    print("f score ",f)
    print("pvalue ",pval)
    
    ''' NOT BEING USED
    ## Feature Scaling --> Standardisation using min max scaler
    from sklearn.preprocessing import MinMaxScaler, StandardScaler
    scale = MinMaxScaler()
    data_scale = scale.fit_transform(data_num_imputed)
    data_scale = pd.DataFrame(data_scale)
    test_scale = scale.transform(test_num_imputed)
    test_scale = pd.DataFrame(test_scale)
    
    ## Joining the data
    data_prepared2 = pd.concat([data_scale,data_vect2],axis=1)
    test_data2 = pd.concat([test_scale,test_vect2],axis=1)
    '''
    # returning vectorised data
    return data_vect2, test_vect2, labels


##  4 layered neural networks model
def create_model(dim):
    model = Sequential()
    model.add(Dense(32, input_dim=dim, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(1, activation="linear"))
    return model


    
if __name__ == "__main__":
    '''
    #laptop path
    data = pd.read_csv(r"E:\Intelligent\tcd ml 2019-20 income prediction training (with labels).csv")
    test = pd.read_csv(r"E:\Intelligent\tcd ml 2019-20 income prediction test (without labels).csv")
    '''
    # desktop path
    data = pd.read_csv(r"U:\datasetML\tcd ml 2019-20 income prediction training (with labels).csv")
    test = pd.read_csv(r"U:\datasetML\tcd ml 2019-20 income prediction test (without labels).csv")

    #Calling the data_prep function
    data_prepared2, test_data2, labels  = data_prep(data,test)
    
    print("Outside Preprocessing")    
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import train_test_split
    
    X_train, X_test, Y_train, Y_test = train_test_split(data_prepared2, labels, test_size = 0.2, random_state = 0)
   
    ## removing the labels from columns
    X_train = X_train.values
    X_test = X_test.values
    Y_train = Y_train.values
    Y_test = Y_test.values
    
    
    
    ######### Neural Networks ################
    model = create_model(X_train.shape[1])
    opt = Adam(lr=1e-2, decay=1e-3 / 200)
    model.compile(loss="mean_squared_error", optimizer=opt)
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),epochs=200, batch_size=100)
    Y_pred = model.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("Neural Networks score is {}".format(np.sqrt(score)))
    
    '''
    ########### Linear Regression ###########
    from sklearn.linear_model import LinearRegression
    lin = LinearRegression()
    lin.fit(X_train,Y_train)
    Y_pred = lin.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("Linear Regression score is {}".format(np.sqrt(score)))
    
    
    ########## SGD regressor ###############
    from sklearn.linear_model import SGDRegressor
    sgd = SGDRegressor()
    sgd.fit(X_train,Y_train)
    Y_pred = sgd.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("SGDregressor score is ",np.sqrt(score))
    '''
    
    ########## Ridge Regression ########
    from sklearn.linear_model import Ridge
    rid = Ridge(alpha=1, solver="cholesky",max_iter=1000)
    rid.fit(X_train,Y_train)
    Y_pred = rid.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("Ridge score is ",np.sqrt(score))
    
    '''
    ###### Random forest regression ######    
    from sklearn.ensemble import RandomForestRegressor
    ran = RandomForestRegressor()
    ran.fit(X_train,Y_train)
    Y_pred=ran.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("Random forest score is {}".format(np.sqrt(score)))
    '''
    '''
    ########## LassoCV Regression ########
    from sklearn.linear_model import LassoCV
    las = LassoCV(alphas=[0.0005, 0.1, 0.001, 1])
    las.fit(X_train,Y_train)
    Y_pred = las.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("Lasso score is ",np.sqrt(score))
    
    
    ########## Elastic Regression ##########
    from sklearn.linear_model import ElasticNet
    elas = ElasticNet(alpha=0.1, l1_ratio=0.5)
    elas.fit(X_train,Y_train)
    Y_pred = elas.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("Elastic score is ",np.sqrt(score))
    
    
    ############## XGBoost ###############
    import xgboost as xgb
    xgb_reg = xgb.XGBRegressor()
    xgb_reg.fit(X_train,Y_train)
    Y_pred = xgb_reg.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("XGBoost score is ",np.sqrt(score))
    
    
    ##########  Light GBM #################
    import lightgbm as lgb
    d2 = lgb.Dataset(X_train, label=Y_train)
    d3 = lgb.Dataset(X_test)
    param = {'objective': 'regression_l2','metric':'rmse','num_leaves':100}
    num_round = 100
    bst = lgb.train(param, d2, num_round, valid_sets=[d3])
    Y_pred = bst.predict(X_test)
    score = mean_squared_error(Y_test,Y_pred)
    print("LGBM score is ",np.sqrt(score))
    '''
    
    ## Writing to a file
    nos = np.empty((73230,1))
    for i in range(73230):
        nos[i] = i+111994
        
    pred = model.predict(test_data2)
    nos = pd.DataFrame(nos)
    pred = pd.DataFrame(pred)
    final = pd.concat([nos,pred],axis=1)
    result = final.to_csv('Submission_Neural.csv')
     
    
