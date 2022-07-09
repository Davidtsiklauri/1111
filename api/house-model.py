import pandas as pd # data processing
from sklearn.model_selection import train_test_split # data split

from sklearn.linear_model import LinearRegression # OLS algorithm
from sklearn.linear_model import Ridge # Ridge algorithm
from sklearn.linear_model import Lasso # Lasso algorithm
from sklearn.linear_model import BayesianRidge # Bayesian algorithm
from sklearn.linear_model import ElasticNet # ElasticNet algorithm
from sklearn.ensemble import RandomForestRegressor

from joblib import dump



# IMPORTING DATA

df = pd.read_csv('./data/house-train.csv')
df.set_index('Id', inplace = True)

df.dropna(inplace = True)


X_var = df[['lat','long','balcony','loggia','veranda','kitchen','numberOfFloors','floor','numberOfRooms','numberOfBedrooms','LotArea','ceilingHeight','numberOfBathrooms']].values
y_var = df['SalePrice'].values

X_train, X_test, y_train, y_test = train_test_split(X_var, y_var, test_size = 0.2, random_state = 0)


# 1. OLS

ols = LinearRegression()
ols.fit(X_train, y_train)
ols_yhat = ols.predict(X_test)

# 2. Ridge

ridge = Ridge(alpha = 0.5)
ridge.fit(X_train, y_train)
ridge_yhat = ridge.predict(X_test)

# 3. Lasso

lasso = Lasso(alpha = 0.01)
lasso.fit(X_train, y_train)
lasso_yhat = lasso.predict(X_test)

# 4. Bayesian

bayesian = BayesianRidge()
bayesian.fit(X_train, y_train)
bayesian_yhat = bayesian.predict(X_test)

# 5. ElasticNet

en = ElasticNet(alpha = 0.01)
en.fit(X_train, y_train)
en_yhat = en.predict(X_test)

# FOREST   
forest = RandomForestRegressor()
forest.fit(X_train, y_train)
forst_yhat= forest.predict(X_test)


dump(ols, "./model_weights/ols.bin")
dump(ridge, "./model_weights/ridge.bin")
dump(lasso, "./model_weights/lasso.bin")
dump(bayesian, "./model_weights/bayesian.bin")
dump(en, "./model_weights/en.bin")
dump(forest, "./model_weights/forest.bin")
