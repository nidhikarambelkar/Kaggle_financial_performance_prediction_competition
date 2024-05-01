# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xg

# Load the train and test datasets
train=pd.read_csv("train.csv", index_col=False)
test=pd.read_csv("test.csv", index_col=False)

# Print the shape of the datasets and columns of the test dataset
print(train.shape)
print(test.shape)
print(test.columns)

# Check for null values in the train dataset
nullcheck=train.isnull().sum()
nullcheck=nullcheck.sort_values(ascending=False)
lissst=list(nullcheck.index)
print(lissst)

# Initialize the LabelEncoder
lb=LabelEncoder()

# Encode categorical variables in the test dataset
for i in test.columns:
  if test[i].dtype=='object':
    test[i]=lb.fit_transform(test[i])

# Encode categorical variables in the train dataset
for i in train.columns:
  if train[i].dtype=='object':
    train[i]=lb.fit_transform(train[i])

# Calculate correlation of the train dataset
correlation=train.corr()
correlation_q0=correlation.loc['Q0_TOTAL_ASSETS': 'Q0_EBITDA', : ]
correlation_mean=correlation_q0.mean(axis=0).sort_values(ascending=False)

# Create a dataframe to compare correlation and null values
compare_dataframe=pd.DataFrame([])
hmm=list(correlation_mean.index)
for i in hmm:
    print(i)
    compare_dataframe.loc[i, 'Correlation']=correlation_mean[i]
    compare_dataframe.loc[i, 'Nullvalues']=nullcheck[i]

# Sort the dataframe by null values
compare_dataframe=compare_dataframe.sort_values(by='Nullvalues', ascending=False)
print(list(compare_dataframe.index))

# Check for null values in the train dataset again
nullcheck2=train.isnull().sum()
print(nullcheck2)

# Print the first 10 rows of the train dataset
train.head(10)


# Drop certain columns from the train and test datasets
# Columns with too many null values are dropped
# Applying run test to check randomness in the data-remove columns with no randomness
# from statsmodels.sandbox.stats.runs import runstest_1samp 
# l1=[]
# for i in list(test.columns):
#     r=runstest_1samp(train[i], correction=False)
#     if r[1]<0.0334:
#         l1.append(i)        
# print(l1)
# Applied this code after imputation and we got'Q4_COST_OF_REVENUES', 'Q5_COST_OF_REVENUES', 'Q6_COST_OF_REVENUES' with no randomness
#but we have dropped it here.
train=train.drop(['trailingPE', 'overallRisk', 'auditRisk',
'boardRisk', 'compensationRisk', 'shareHolderRightsRisk',
'Q7_NET_INCOME', 'Q10_COST_OF_REVENUES','Q3_NET_INCOME',
'Q10_REVENUES','Q7_COST_OF_REVENUES', 'Q7_REVENUES',
'Q2_NET_INCOME','Q1_NET_INCOME', 'Q9_COST_OF_REVENUES', 'Q9_REVENUES',
"Q10_fiscal_year_end",'Q9_fiscal_year_end','Q8_fiscal_year_end',
'Q6_fiscal_year_end','Q5_fiscal_year_end','Q4_fiscal_year_end',
'Q1_fiscal_year_end','financialCurrency','Q1_DEPRECIATION_AND_AMORTIZATION',
'Q2_DEPRECIATION_AND_AMORTIZATION','Q4_COST_OF_REVENUES','targetMeanPrice', 
'Q4_COST_OF_REVENUES', 'Q5_COST_OF_REVENUES', 'Q6_COST_OF_REVENUES',
 'Q7_OPERATING_INCOME', 'Q2_fiscal_year_end', 'Q3_fiscal_year_end',
 'recommendationMean', 'recommendationKey'], axis=1)
test=test.drop(['trailingPE', 'overallRisk', 'auditRisk',
'boardRisk', 'compensationRisk', 'shareHolderRightsRisk',
'Q7_NET_INCOME', 'Q10_COST_OF_REVENUES','Q3_NET_INCOME',
'Q10_REVENUES','Q7_COST_OF_REVENUES', 'Q7_REVENUES',
'Q2_NET_INCOME','Q1_NET_INCOME', 'Q9_COST_OF_REVENUES', 'Q9_REVENUES',
"Q10_fiscal_year_end",'Q9_fiscal_year_end','Q8_fiscal_year_end',
'Q6_fiscal_year_end','Q5_fiscal_year_end','Q4_fiscal_year_end',
'Q1_fiscal_year_end','financialCurrency','Q1_DEPRECIATION_AND_AMORTIZATION',
'Q2_DEPRECIATION_AND_AMORTIZATION','Q4_COST_OF_REVENUES','targetMeanPrice', 
'Q4_COST_OF_REVENUES', 'Q5_COST_OF_REVENUES', 'Q6_COST_OF_REVENUES',
 'Q7_OPERATING_INCOME', 'Q2_fiscal_year_end', 'Q3_fiscal_year_end', 
 'recommendationMean', 'recommendationKey'], axis=1)

# Initialize the SimpleImputer
impute2=SimpleImputer(missing_values=np.nan,strategy="mean")

# Check for infinity or large values in the train dataset and handle them
problematic_cols = []
for col in train.columns:
    if np.any(np.isinf(train[col])) or np.any(np.abs(train[col]) > 1e15):
        problematic_cols.append(col)
for col in problematic_cols:
    train[col].replace([np.inf, -np.inf], [3.4028235e+38, -3.4028235e+38], inplace=True)

# Impute missing values in the train dataset
for i in train.columns:
    train[i] = impute2.fit_transform(train[i].values.reshape(-1,1))

# Check for infinity or large values in the test dataset and handle them
problematic_cols = []
for col in test.columns:
    if np.any(np.isinf(test[col])) or np.any(np.abs(test[col]) > 1e15):
        problematic_cols.append(col)
for col in problematic_cols:
    test[col].replace([np.inf, -np.inf], [3.4028235e+38, -3.4028235e+38], inplace=True)

# Impute missing values in the test dataset
for i in test.columns:
    test[i] = impute2.fit_transform(test[i].values.reshape(-1,1))

# Calculate variance of the train dataset
std=train.var().sort_values()
print(list(std.index)) 

# Calculate correlation of the train dataset
corr=train.corr()
high_corr_var=np.where(corr>0.98)
high_corr_var=[(corr.columns[x],corr.columns[y]) for x,y in zip(*high_corr_var) if x!=y and x<y]
high_corr_var

# Define dependent variables
dependent='Q0_REVENUES,Q0_COST_OF_REVENUES,Q0_GROSS_PROFIT,Q0_OPERATING_EXPENSES,Q0_EBITDA,Q0_OPERATING_INCOME,Q0_TOTAL_ASSETS,Q0_TOTAL_LIABILITIES,Q0_TOTAL_STOCKHOLDERS_EQUITY'
dependent_list=dependent.split(',')
print(dependent_list)

# Define independent variables
independent_list=[]
for i in test.columns:
  if(i in dependent_list):
    continue
  else:
    independent_list.append(i)
independent_list

# Define training and testing datasets
X_train = train[independent_list]
X_test=test[independent_list]

# Define target variables
Y_train_1=train[['Q0_TOTAL_ASSETS']]
Y_train_2=train[['Q0_TOTAL_LIABILITIES']]
Y_train_3=train[['Q0_TOTAL_STOCKHOLDERS_EQUITY']]
Y_train_4=train[['Q0_GROSS_PROFIT']]
Y_train_5=train[['Q0_COST_OF_REVENUES']]
Y_train_5=train[['Q0_REVENUES']]
Y_train_6=train[['Q0_OPERATING_INCOME']]
Y_train_7=train[['Q0_OPERATING_EXPENSES']]
Y_train_8=train[['Q0_EBITDA']]

# Initialize the XGBoost Regressor
xgr=xg.XGBRegressor()
xgr1=xg.XGBRegressor()

# Create a dataframe to store the predictions
predicted_data1 = pd.DataFrame(test['Id'], index=X_test.index)

# Train the model and make predictions for each dependent variable
for i in dependent_list:
    xgr.fit(X_train, train[[i]])
    from sklearn.feature_selection import SelectFromModel
    selection = SelectFromModel(xgr, threshold=0.00106, prefit=True)
    selected_dataset = selection.transform(X_train)
    selected_dataset2 = selection.transform(X_test)
    xgr1.fit(selected_dataset, train[i])
    predicted_data1[i] = xgr1.predict(selected_dataset2)

# Set the index of the predicted data
predicted_data1=predicted_data1.set_index('Id')

# Save the predictions to a CSV file
predicted_data1.to_csv('submission.csv')
