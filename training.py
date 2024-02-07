# Import necessary libraries
from sqlalchemy import create_engine
from sqlalchemy import text
import pandas as pd
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

# Connect to the PostgreSQL database
engine = create_engine('postgresql://avnadmin:AVNS_jc1VkZHPVPIz6hSz7n7@metlife-mlchallenge.a.aivencloud.com:23357/mlchallenge?sslmode=require')

# Read data from the 'training_dataset' table of the database and load into DataFrame 'dfCustomers'
dfCustomers = pd.read_sql_query(text('SELECT age, sex, bmi, children, smoker, region, charges FROM training_dataset'), engine.connect())

# Separation of columns into variable types
numeric = ['age', 'bmi',  'children']
categorical = ['sex', 'smoker', 'region']  # Although 'sex' and 'smoker' are binary, they are treated as categorical variables for One-Hot encoding

# Define the column transformer
transformer = make_column_transformer(
    (MinMaxScaler(), numeric),  # Scale numeric variables
    (OneHotEncoder(), categorical)  # Encode categorical variables
)

# Define the pipeline for the RandomForestRegressor model
rf_estimators = [('column_transformer', transformer), ('clf', RandomForestRegressor(random_state=42))]
pipe_rf = Pipeline(rf_estimators)

# Define the parameter grid for the hyperparameter search of RandomForestRegressor
param_grid_rf = {
    'clf__n_estimators': [100, 200, 300, 400, 500],
    'clf__max_depth': [3, 5, 10, 15, 25, None]
}

# Perform hyperparameter search for RandomForestRegressor
model_gscv_rf = GridSearchCV(pipe_rf, param_grid=param_grid_rf, cv=3, scoring='neg_root_mean_squared_error', verbose=10, n_jobs=-1)
model_gscv_rf.fit(dfCustomers.drop('charges', axis=1), dfCustomers['charges'])

# Define the pipeline for the GradientBoostingRegressor model
gbr_estimators = [('column_transformer', transformer), ('gbr', GradientBoostingRegressor())]
pipe_gbr = Pipeline(gbr_estimators)

# Define the parameter grid for the hyperparameter search of GradientBoostingRegressor
param_grid_gbr = {
    'gbr__n_estimators': [50, 100, 200],
    'gbr__learning_rate': [0.01, 0.05, 0.1],
    'gbr__max_depth': [2, 3, 4]
}

# Perform hyperparameter search for GradientBoostingRegressor
model_gscv_gbr = GridSearchCV(pipe_gbr, param_grid=param_grid_gbr, cv=3, scoring='neg_root_mean_squared_error', verbose=0, n_jobs=-1)
model_gscv_gbr.fit(dfCustomers.drop('charges', axis=1), dfCustomers['charges'])


import pickle

def save_best_model(rf_model, gbr_model, pickle_path, txt_path):
    # Determine the model with the best score closest to zero
    if abs(rf_model.best_score_) < abs(gbr_model.best_score_):
        best_model = rf_model
        best_params = rf_model.best_params_
        best_score = abs(rf_model.best_score_)
        best_estimator = rf_model.best_estimator_
        model_name = "Random Forest"
    else:
        best_model = gbr_model
        best_params = gbr_model.best_params_
        best_score = abs(gbr_model.best_score_)
        best_estimator = gbr_model.best_estimator_
        model_name = "Gradient Boosting"

    # Save the best model to a pickle file
    with open(pickle_path, 'wb') as pickle_file:
        pickle.dump(best_model, pickle_file)

    # Save the best parameters and best estimator to a text file
    with open(txt_path, 'w') as txt_file:
        txt_file.write(f"Best model: {model_name}\n")
        txt_file.write(f"Best score: {best_score}\n")
        txt_file.write("Best parameters:\n")
        for key, value in best_params.items():
            txt_file.write(f"{key}: {value}\n")
        txt_file.write(f"Best estimator:\n{best_estimator}")


save_best_model(model_gscv_rf, model_gscv_gbr, 'model.pkl', 'model_desc.txt')

print('Successful training. Model and its characteristics saved.')
