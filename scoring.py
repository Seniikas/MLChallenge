# Import necessary libraries
import pickle
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy import text

# Connect to the PostgreSQL database
engine = create_engine('postgresql://avnadmin:AVNS_jc1VkZHPVPIz6hSz7n7@metlife-mlchallenge.a.aivencloud.com:23357/mlchallenge?sslmode=require')

# Read data from the 'training_dataset' table of the database and load into DataFrame 'dfPredict'
dfPredict = pd.read_sql_query(text('SELECT age, sex, bmi, children, smoker, region FROM training_dataset ORDER BY RANDOM() LIMIT 10'), engine.connect())

# Load the saved model
with open('model.pkl', 'rb') as file:
    model = pickle.load(file)

# Predict the loaded dataset
pred_charges = model.predict(dfPredict)

# Add the prediction results to the DataFrame
dfPredict['pred_charges'] = pred_charges

# Store the predictions in a new table
dfPredict.to_sql('predictions', engine, if_exists='replace')

print('Prediction successfully made')

engine.dispose()
