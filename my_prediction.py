import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import make_column_selector, make_column_transformer
from sklearn.model_selection import  train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import GridSearchCV

data_train = pd.read_csv('train.csv')
data_test = pd.read_csv('test.csv')
pd.set_option('display.max_rows', None)

# print(data.isna().sum())
# print(data.shape) # 1460, 81 
# print(data.columns)

data_train = data_train.drop(columns=['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature'])
data_test = data_test.drop(columns=['Alley', 'MasVnrType', 'PoolQC', 'Fence', 'MiscFeature'])

#print(data.dtypes)# integers, objects

y = data_train['SalePrice']
X = data_train.drop(columns=['SalePrice', 'Id'])
id_column = data_test.pop('Id')

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)


num_pipe = make_pipeline(SimpleImputer(strategy='median'))
cat_pipe= make_pipeline(SimpleImputer(strategy='most_frequent'), 
                        OneHotEncoder(handle_unknown='ignore', sparse_output=False))



ct = make_column_transformer(
    (num_pipe, make_column_selector(dtype_include='number')),
    (cat_pipe, make_column_selector(dtype_include='object'))         
)
ct.set_output(transform='pandas')


final_pipeline = make_pipeline(ct, RandomForestRegressor())

param_grid = {
    'randomforestregressor__max_depth': range(2,15,2)
}
my_search = GridSearchCV(
    final_pipeline,
    param_grid=param_grid,
    cv=5,
    verbose=1,
    n_jobs=-1,
    scoring= 'neg_root_mean_squared_log_error'
)
my_search.fit(X_train, y_train)

my_model = my_search.best_estimator_

X = data_train.drop('Id',axis=1)
y = X.pop('SalePrice')

my_model.fit(X,y)
pred = my_model.predict(data_test)

results = pd.DataFrame({
    'Id':id_column,
    'SalePrice':pred
})

results.to_csv('submission.csv', index=False)
print("Saved to submission.csv")


