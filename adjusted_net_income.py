# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 23:12:48 2016

@author: Bernard
"""

#from pandas import Series, DataFrame
import pandas as pd
import numpy as np
import scipy
import seaborn as sns
import matplotlib.pylab as plt
from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LassoLarsCV


 
#Load the dataset
data = pd.read_csv("worldbank.csv")


###############################################################################
# DATA MANAGEMENT
# CONVERTING SOME QUANTITATIVE VARIABLES INTO CATEGORICAL VARIABLES TO GET FEEL 
# OF LASSO REGRESSION
###############################################################################


data_clean = data.dropna()





###############################################################################
#END OF DATA MANAGEMENT
###############################################################################



predvar= data_clean[['x1_2012', 'x2_2012', 'x9_2012', 'x12_2012',
    'x14_2012', 'x15_2012', 'x16_2012', 'x18_2012', 'x19_2012', 'x21_2012',
   'x25_2012',	'x29_2012', 'x31_2012', 'x35_2012', 'x36_2012', 'x37_2012',
   'x38_2012', 'x45_2012', 'x47_2012', 'x48_2012', 'x49_2012', 'x58_2012',
   'x67_2012', 'x68_2012', 'x69_2012', 'x86_2012', 'x100_2012', 'x121_2012',
   'x125_2012', 'x126_2012', 'x129_2012', 'x131_2012',  'x132_2012',
   'x134_2012',  'x139_2012',  'x140_2012', 'x142_2012', 'x143_2012',
   'x146_2012', 'x149_2012', 'x150_2012', 'x153_2012', 'x154_2012', 'x155_2012',
   'x156_2012', 'x157_2012',  'x161_2012', 'x162_2012',   'x163_2012',
   'x167_2012', 'x169_2012', 'x171_2012', 'x172_2012', 'x173_2012', 'x174_2012',
   'x179_2012', 'x187_2012', 'x190_2012',    'x191_2012', 'x192_2012', 
   'x195_2012', 'x204_2012', 'x205_2012', 'x211_2012', 'x212_2012', 'x213_2012', 
   'x218_2012', 'x219_2012', 'x220_2012', 'x221_2012', 'x222_2012',  'x223_2012', 
   'x242_2012', 'x243_2012', 'x244_2012', 'x253_2012',  'x255_2012', 'x258_2012', 
   'x261_2012', 'x268_2012', 'x274_2012', 'x275_2012',  'x277_2012', 
   'x283_2012','x284_2012']]

#check length of colunns in predictor variables
print("column length of predvar = ",len(predvar.columns))

#check the number of rows of the entire predicctor variables
print ("row length of predvar = ", len(predvar))


target = data_clean.x11_2012



 
# standardize predictors to have mean=0 and sd=1
predictors=predvar.copy()
from sklearn import preprocessing
predictors['x1_2012']=preprocessing.scale(predictors['x1_2012'].astype('float64'))
predictors['x2_2012']=preprocessing.scale(predictors['x2_2012'].astype('float64'))
predictors['x9_2012']=preprocessing.scale(predictors['x9_2012'].astype('float64'))
predictors['x12_2012']=preprocessing.scale(predictors['x12_2012'].astype('float64'))
predictors['x14_2012']=preprocessing.scale(predictors['x14_2012'].astype('float64'))
predictors['x15_2012']=preprocessing.scale(predictors['x15_2012'].astype('float64'))
predictors['x16_2012']=preprocessing.scale(predictors['x16_2012'].astype('float64'))
predictors['x18_2012']=preprocessing.scale(predictors['x18_2012'].astype('float64'))
predictors['x19_2012']=preprocessing.scale(predictors['x19_2012'].astype('float64'))
predictors['x21_2012']=preprocessing.scale(predictors['x21_2012'].astype('float64'))
predictors['x25_2012']=preprocessing.scale(predictors['x25_2012'].astype('float64'))
predictors['x29_2012']=preprocessing.scale(predictors['x29_2012'].astype('float64'))
predictors['x31_2012']=preprocessing.scale(predictors['x31_2012'].astype('float64'))
predictors['x35_2012']=preprocessing.scale(predictors['x35_2012'].astype('float64'))
predictors['x36_2012']=preprocessing.scale(predictors['x36_2012'].astype('float64'))
predictors['x37_2012']=preprocessing.scale(predictors['x37_2012'].astype('float64'))
predictors['x38_2012']=preprocessing.scale(predictors['x38_2012'].astype('float64'))
predictors['x45_2012']=preprocessing.scale(predictors['x45_2012'].astype('float64'))
predictors['x47_2012']=preprocessing.scale(predictors['x47_2012'].astype('float64'))
predictors['x48_2012']=preprocessing.scale(predictors['x48_2012'].astype('float64'))
predictors['x49_2012']=preprocessing.scale(predictors['x49_2012'].astype('float64'))
predictors['x58_2012']=preprocessing.scale(predictors['x58_2012'].astype('float64'))
predictors['x67_2012']=preprocessing.scale(predictors['x67_2012'].astype('float64'))
predictors['x68_2012']=preprocessing.scale(predictors['x68_2012'].astype('float64'))
predictors['x69_2012']=preprocessing.scale(predictors['x69_2012'].astype('float64'))
predictors['x86_2012']=preprocessing.scale(predictors['x86_2012'].astype('float64'))
predictors['x100_2012']=preprocessing.scale(predictors['x100_2012'].astype('float64'))
predictors['x121_2012']=preprocessing.scale(predictors['x121_2012'].astype('float64'))
predictors['x125_2012']=preprocessing.scale(predictors['x125_2012'].astype('float64'))
predictors['x126_2012']=preprocessing.scale(predictors['x126_2012'].astype('float64'))
predictors['x129_2012']=preprocessing.scale(predictors['x129_2012'].astype('float64'))
predictors['x131_2012']=preprocessing.scale(predictors['x131_2012'].astype('float64'))
predictors['x132_2012']=preprocessing.scale(predictors['x132_2012'].astype('float64'))
predictors['x134_2012']=preprocessing.scale(predictors['x134_2012'].astype('float64'))
predictors['x139_2012']=preprocessing.scale(predictors['x139_2012'].astype('float64'))
predictors['x140_2012']=preprocessing.scale(predictors['x140_2012'].astype('float64'))
predictors['x142_2012']=preprocessing.scale(predictors['x142_2012'].astype('float64'))
predictors['x143_2012']=preprocessing.scale(predictors['x143_2012'].astype('float64'))
predictors['x146_2012']=preprocessing.scale(predictors['x146_2012'].astype('float64'))
predictors['x149_2012']=preprocessing.scale(predictors['x149_2012'].astype('float64'))
predictors['x150_2012']=preprocessing.scale(predictors['x150_2012'].astype('float64'))
predictors['x153_2012']=preprocessing.scale(predictors['x153_2012'].astype('float64'))
predictors['x154_2012']=preprocessing.scale(predictors['x154_2012'].astype('float64'))
predictors['x155_2012']=preprocessing.scale(predictors['x155_2012'].astype('float64'))
predictors['x156_2012']=preprocessing.scale(predictors['x156_2012'].astype('float64'))
predictors['x157_2012']=preprocessing.scale(predictors['x157_2012'].astype('float64'))
predictors['x161_2012']=preprocessing.scale(predictors['x161_2012'].astype('float64'))
predictors['x162_2012']=preprocessing.scale(predictors['x162_2012'].astype('float64'))
predictors['x163_2012']=preprocessing.scale(predictors['x163_2012'].astype('float64'))
predictors['x167_2012']=preprocessing.scale(predictors['x167_2012'].astype('float64'))
predictors['x169_2012']=preprocessing.scale(predictors['x169_2012'].astype('float64'))
predictors['x171_2012']=preprocessing.scale(predictors['x171_2012'].astype('float64'))
predictors['x172_2012']=preprocessing.scale(predictors['x172_2012'].astype('float64'))
predictors['x173_2012']=preprocessing.scale(predictors['x173_2012'].astype('float64'))
predictors['x174_2012']=preprocessing.scale(predictors['x174_2012'].astype('float64'))
predictors['x179_2012']=preprocessing.scale(predictors['x179_2012'].astype('float64'))
predictors['x187_2012']=preprocessing.scale(predictors['x187_2012'].astype('float64'))
predictors['x190_2012']=preprocessing.scale(predictors['x190_2012'].astype('float64'))
predictors['x191_2012']=preprocessing.scale(predictors['x191_2012'].astype('float64'))
predictors['x192_2012']=preprocessing.scale(predictors['x192_2012'].astype('float64'))
predictors['x195_2012']=preprocessing.scale(predictors['x195_2012'].astype('float64'))
predictors['x204_2012']=preprocessing.scale(predictors['x204_2012'].astype('float64'))
predictors['x205_2012']=preprocessing.scale(predictors['x205_2012'].astype('float64'))
predictors['x211_2012']=preprocessing.scale(predictors['x211_2012'].astype('float64'))
predictors['x212_2012']=preprocessing.scale(predictors['x212_2012'].astype('float64'))
predictors['x213_2012']=preprocessing.scale(predictors['x213_2012'].astype('float64'))
predictors['x218_2012']=preprocessing.scale(predictors['x218_2012'].astype('float64'))
predictors['x219_2012']=preprocessing.scale(predictors['x219_2012'].astype('float64'))
predictors['x220_2012']=preprocessing.scale(predictors['x220_2012'].astype('float64'))
predictors['x221_2012']=preprocessing.scale(predictors['x221_2012'].astype('float64'))
predictors['x222_2012']=preprocessing.scale(predictors['x222_2012'].astype('float64'))
predictors['x223_2012']=preprocessing.scale(predictors['x223_2012'].astype('float64'))
predictors['x242_2012']=preprocessing.scale(predictors['x242_2012'].astype('float64'))
predictors['x243_2012']=preprocessing.scale(predictors['x243_2012'].astype('float64'))
predictors['x244_2012']=preprocessing.scale(predictors['x244_2012'].astype('float64'))
predictors['x253_2012']=preprocessing.scale(predictors['x253_2012'].astype('float64'))
predictors['x255_2012']=preprocessing.scale(predictors['x255_2012'].astype('float64'))
predictors['x258_2012']=preprocessing.scale(predictors['x258_2012'].astype('float64'))
predictors['x261_2012']=preprocessing.scale(predictors['x261_2012'].astype('float64'))
predictors['x268_2012']=preprocessing.scale(predictors['x268_2012'].astype('float64'))
predictors['x274_2012']=preprocessing.scale(predictors['x274_2012'].astype('float64'))
predictors['x275_2012']=preprocessing.scale(predictors['x275_2012'].astype('float64'))
predictors['x277_2012']=preprocessing.scale(predictors['x277_2012'].astype('float64'))
predictors['x283_2012']=preprocessing.scale(predictors['x283_2012'].astype('float64'))
predictors['x284_2012']=preprocessing.scale(predictors['x284_2012'].astype('float64'))


#check if predictors for lasso regression are standardized and have mean=0 and 
#sd=1 the means, standard deviation
for stats in predictors:
    print(predictors[stats].describe())

# split data into train and test sets
pred_train, pred_test, tar_train, tar_test = train_test_split(predictors, target, 
                                                              test_size=.3, random_state=123)
                                            


# specify the lasso regression model
model=LassoLarsCV(cv=10, precompute=False).fit(pred_train,tar_train)

# print variable names and regression coefficients
dict(zip(predictors.columns, model.coef_))

#dicionary of predictor variables retained in the model
dictionaryValues = dict(zip(predictors.columns, model.coef_))


#make copy of predictors to work towards predictors which were retained during the lasso regression
predictorsNew =    predictors.copy()  

#list to contain the retained values in Lasso Regression  
listofVals = []
for predictorsNew, v in dictionaryValues.items():
    if v != 0.0:
       print(v)
       listofVals.append(predictorsNew)

       
#list of predictors retained in the Lasso Regression     
print(listofVals) 

#print predictors retained in the model and their associattive co-efficients
print('predictor , co-efficient')
for predictorsNew, v in dictionaryValues.items():
    if v != 0.0:
       print(predictorsNew, v)

#assessing list of retained values from the dataframe
predictorsNew = predvar[listofVals]
print("length and list of new preds",predictorsNew.columns)

#check the means, standard deviation, maximum and minimum of the retained variables
for stats in predictorsNew:
    print(predictorsNew[stats].describe())
    

#check the means, standard deviation, maximum and minimum of the response variables 
data_clean["x11_2012"].describe()




# convert quantitative predictor variables to numeric format
for numericformat in predictorsNew:
    print(numericformat)
    predictorsNew[numericformat] = pd.to_numeric(predictorsNew[numericformat], errors='coerce')

# convert quantitative predictor variables in data_clean to numeric format
for numericformat in data_clean:
    print(numericformat)
    data_clean[numericformat] = pd.to_numeric(data_clean[numericformat], errors='coerce')

# convert quantitative response variables to numeric format
data_clean["x11_2012"] =pd.to_numeric(data_clean["x11_2012"], errors='coerce')


#Generating univariate histogram for the quantitative variaables
#x218_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x218_2012"].dropna(), kde=False);
plt.xlabel('x218_2012')
plt.title('Predictor variable x218_2012')

    
#x275_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x275_2012"].dropna(), kde=False);
plt.xlabel('x275_2012')
plt.title('Predictor variable x275_2012')
    
#x261_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x261_2012"].dropna(), kde=False);
plt.xlabel('x261_2012')
plt.title('Predictor variable x261_2012')

    
#x142_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x142_2012"].dropna(), kde=False);
plt.xlabel('x142_2012')
plt.title('Predictor variable x142_2012')

#x258_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x258_2012"].dropna(), kde=False);
plt.xlabel('x258_2012')
plt.title('Predictor variable x258_2012')

#x126_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x126_2012"].dropna(), kde=False);
plt.xlabel('x126_2012')
plt.title('Predictor variable x126_2012')

#x121_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x121_2012"].dropna(), kde=False);
plt.xlabel('x121_2012')
plt.title('Predictor variable x121_2012')

#x12_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x12_2012"].dropna(), kde=False);
plt.xlabel('x12_2012')
plt.title('Predictor variable x12_2012')

#x149_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x149_2012"].dropna(), kde=False);
plt.xlabel('x149_2012')
plt.title('Predictor variable x149_2012')

#x21_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x21_2012"].dropna(), kde=False);
plt.xlabel('x21_2012')
plt.title('Predictor variable x21_2012')

#x9_2012 Univariate histogram for quantitative variable:
sns.distplot(predictorsNew["x9_2012"].dropna(), kde=False);
plt.xlabel('x9_2012')
plt.title('Predictor variable x9_2012')
    
#x11_2012 Univariate histogram for quantitative response variable:
sns.distplot(data_clean["x11_2012"].dropna(), kde=False);
plt.xlabel('x11_2012')
plt.title('response variable x11_2012')



#print frequency values for predictor variables
for stats in predictorsNew:
    print(predictorsNew[stats].value_counts(sort = False ,  dropna = False))


# split data for scatterplots
train,test=train_test_split(data_clean, test_size=.4, random_state=123)




#scatterplots for first 5 of  the quantitative predictor and quantitative response variable
fig1 = sns.PairGrid(train, y_vars=["x11_2012"], 
                 x_vars= listofVals[:5],\
                 size=5, palette="GnBu_d")
fig1.map(plt.scatter, s=50, edgecolor="white")
plt.title('Figure 3. Association Between Quantitative Predictors and x11_2012 - Adjusted Net National Income Per Capita ,(Current Us$) ', 
                    fontsize = 12, loc='right')
fig1.savefig('reportfig1.jpg')

#scatterplots for first second set of  the quantitative predictor and quantitative response variable
fig2 = sns.PairGrid(train, y_vars=["x11_2012"], 
                 x_vars= listofVals[5:],\
                 size=5, palette="GnBu_d")
fig2.map(plt.scatter, s=50, edgecolor="white")
plt.title('Figure 4. Association Between Quantitative Predictors and x11_2012 - Adjusted Net National Income Per Capita ,(Current Us$) ', 
                    fontsize = 12, loc='right')
fig2.savefig('reportfig1.jpg')


#pearson correlation "r" value and relative p-value for quantitative variables
for corValue in predictorsNew:
    print ('pearson correlation "r" value and relative p-value for the association between', corValue, ' and x11_2012')
    print (scipy.stats.pearsonr(predictorsNew[corValue], data_clean["x11_2012"]))
    print("")


  
# plot coefficient progression
m_log_alphas = -np.log10(model.alphas_)
ax = plt.gca()
plt.plot(m_log_alphas, model.coef_path_.T)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.ylabel('Regression Coefficients')
plt.xlabel('-log(alpha)')
plt.title('Regression Coefficients Progression for Lasso Paths')

# plot mean square error for each fold
m_log_alphascv = -np.log10(model.cv_alphas_)
plt.figure()
plt.plot(m_log_alphascv, model.cv_mse_path_, ':')
plt.plot(m_log_alphascv, model.cv_mse_path_.mean(axis=-1), 'k',
         label='Average across the folds', linewidth=2)
plt.axvline(-np.log10(model.alpha_), linestyle='--', color='k',
            label='alpha CV')
plt.legend()
plt.xlabel('-log(alpha)')
plt.ylabel('Mean squared error')
plt.title('Mean squared error on each fold')
         

# MSE from training and test data
from sklearn.metrics import mean_squared_error
train_error = mean_squared_error(tar_train, model.predict(pred_train))
test_error = mean_squared_error(tar_test, model.predict(pred_test))
print ('training data MSE')
print(train_error)
print ('test data MSE')
print(test_error)

# R-square from training and test data
rsquared_train=model.score(pred_train,tar_train)
rsquared_test=model.score(pred_test,tar_test)
print ('training data R-square')
print(rsquared_train)
print ('test data R-square')
print(rsquared_test)
