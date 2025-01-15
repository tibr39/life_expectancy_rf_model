import kagglehub
import pandas as pd



#Loading datasets
df = pd.read_csv("life_expectancy.csv")
smoking = pd.read_csv("smoking_deathrate.csv")

print(df)
print(smoking)


#making dictionary between different different nomination of countries
country_translation = {
    'Turkey': 'Turkiye',
    'Iran': 'Iran, Islamic Rep.',
    'Syria': 'Syrian Arab Republic',
    'Sudan': 'Sudan',
    'Kyrgyzstan': 'Kyrgyz Republic',
    'North Korea': 'Korea, Dem. People’s Rep.',
    'South Korea': 'Korea, Rep.',
    'Russia': 'Russian Federation',
    'Congo': 'Congo, Rep.',
    'Democratic Republic of Congo': 'Congo, Dem. Rep.',
    'Gambia': 'Gambia, The',
    'Bahamas': 'Bahamas, The',
    'Micronesia (country)': 'Micronesia, Fed. Sts.',
    'Laos': 'Lao PDR',
    'Venezuela': 'Venezuela, RB',
    'Egypt': 'Egypt, Arab Rep.',
    'Cape Verde': 'Cabo Verde',
    'East Timor': 'Timor-Leste',
    'Palestine': 'West Bank and Gaza',
    'Brunei':'Brunei Darussalam',
    'Slovakia':'Slovak Republic',
    'Yemen':'Yemen, Rep.',
    'Dominica':'Dominican Republic',
    'Saint Lucia':'St. Lucia',
    'Saint Vincent and the Grenadines':'St. Vincent and the Grenadines'



}
#replacing country names based on main df dataset
smoking['Country'] = smoking['Country'].replace(country_translation)

#hm = df['Country'].unique()

#print(hm)

#show unmatched countries whihc have no pair in main dataset
unmatched_countries_after_translation = smoking[~smoking['Country'].isin(df['Country'])]
print("Remaining unmatched countries after translation:")
print(unmatched_countries_after_translation['Country'].unique())

#filtering smoking dataset based on main dataset
filtered_smoking = smoking[smoking['Country'].isin(df['Country'])]
print(filtered_smoking)


#filtered_smoking.to_csv("filtered_smoking_deathrate.csv")

#mergin the two dataset based on country and year
merged_df = pd.merge(df, filtered_smoking[['Country', 'Year', 'Value']], on=['Country', 'Year'], how='left')

print(merged_df)

#merged_df.to_csv("complete_dataset.csv")


#Model building


from sklearn.model_selection import train_test_split

# Extract feature and target arrays
X, y = merged_df.drop(['Population_mln',"Region","Country","Year","Life_expectancy"], axis=1), merged_df[['Life_expectancy']]



# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)



import xgboost as xgb


# Create regression matrices
dtrain_reg = xgb.DMatrix(X_train, y_train, enable_categorical=True)
dtest_reg = xgb.DMatrix(X_test, y_test, enable_categorical=True)

import numpy as np

#mse = np.mean((actual - predicted) ** 2)
#rmse = np.sqrt(mse)


# Define hyperparameters
params = {"objective": "reg:squarederror", "tree_method": "hist", "device":"cuda"}

#print(X,y)

n = 100
model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
)

from sklearn.metrics import mean_squared_error
print(model.feature_names)

#test = [45.2, 57.2, 434.821, 7.49, 90, 83, 26.8, 84, 90, 14.3, 3680, 4, 4.1, 6.5, 0, 1, 28.475904]


#test_df = pd.DataFrame([test], columns = model.feature_names)


#preds = model.predict(xgb.DMatrix(test_df))

#preds = model.predict(dtest_reg)


#rmse = mean_squared_error(y_test, preds, squared=False)





params = {"objective": "reg:squarederror",
          'learning_rate': 0.01,  # Lower learning rate
          'max_depth': 6,  # Reduce tree depth to avoid overfitting
          'min_child_weight': 5,  # Prevent small nodes that might overfit
          'subsample': 0.8,  # Use 80% of data for each tree
          'colsample_bytree': 0.8,  # Use 80% of features for each tree
          'reg_alpha': 0.1,  # L1 regularization
          'reg_lambda': 1,  # L2 regularization
          'booster':'gbtree',
          'device' : "cuda"}
n = 20000

evals = [(dtest_reg, "validation"), (dtrain_reg, "train")]


model = xgb.train(
   params=params,
   dtrain=dtrain_reg,
   num_boost_round=n,
   evals=evals,
    early_stopping_rounds=50,
   verbose_eval=250
)
#[19999]	validation-rmse:0.46523	train-rmse:0.01068




n = 20000

results = xgb.cv(
   params, dtrain_reg,
   num_boost_round=n,

   nfold=5,
   early_stopping_rounds=20,
    verbose_eval=250
)

best_rmse = results['test-rmse-mean'].min()

print(best_rmse)

#0.46250978595750214



# Save the trained model
model.save_model('xgboost_model.json')  # Save as JSON