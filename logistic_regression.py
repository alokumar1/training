import pandas as pd
from sklearn.preprocessing import Imputer,LabelEncoder,OneHotEncoder,StandardScaler
import numpy as np
from sklearn.linear_model import LogisticRegression

train = pd.read_csv('train_loan.csv')
test = pd.read_csv('test_loan.csv')
train.head()
def preprocess(train):
    le = LabelEncoder()
    imputer = Imputer(missing_values = 'NaN',strategy='mean',axis = 0)
    train.loc[train.Gender.isnull(),'Gender'] = 'Male'
    train.loc[train.Married.isnull(),'Married'] = 'Yes'
    imputer_new = Imputer(missing_values = 'NaN',strategy='most_frequent',axis = 0)
    train.loc[train.Self_Employed.isnull(),'Self_Employed'] = 'No'
    train.loc[train.Dependents == '3+','Dependents'] = '3'
    train.loc[train.Dependents.isnull(),'Dependents'] = '0'
    train.iloc[:,8:9] = imputer.fit_transform(train.iloc[:,8:9])
    train.Loan_Amount_Term.value_counts().plot('bar')
    train.iloc[:,9:10] = imputer_new.fit_transform(train.iloc[:,9:10])
    train.iloc[:,10:11] = imputer_new.fit_transform(train.iloc[:,10:11])
    train.Gender = le.fit_transform(train.Gender)
    train.Married = le.fit_transform(train.Married)
    train.Education = le.fit_transform(train.Education)
    train.Self_Employed = le.fit_transform(train.Self_Employed)
    train.Property_Area = le.fit_transform(train.Property_Area)
    
    return train
        
train = preprocess(train)            
y = train.Loan_Status
del train['Loan_Status']
del train['Loan_ID']

test = preprocess(test)
test_id = test.Loan_ID
del test['Loan_ID']

lgr = LogisticRegression()
lgr.fit(train,y)
y_pred = lgr.predict(test)
score = lgr.score(train,y)

# frame = pd.DataFrame({'Loan_ID':test_id,'Loan_Status':y_pred})
# frame.to_csv('predictions_modified_8.csv')