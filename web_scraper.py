import pandas as pd
import json
import requests

#Fetching JSON from Our World Data site for Smoking Death Rates



#loading json from website(data mining)
data = requests.get("https://api.ourworldindata.org/v1/indicators/940724.data.json").json()
metadata= requests.get("https://api.ourworldindata.org/v1/indicators/940724.metadata.json").json()



#loading main dataset and the metadata from the website
#data="data.json"
#metadata="metadata.json"


#loading both json file with the json library
with open("data.json", encoding='utf-8', errors='ignore') as json_data:
    data = json.load(json_data, strict=False)

with open("metadata.json", encoding='utf-8', errors='ignore') as json_data:
    metadata = json.load(json_data, strict=False)

#create smoking dataset based on the json file from the website
df = pd.DataFrame({
    'Entity': data['entities'],
    'Year': data['years'],
    'Value': data['values']
})


#extracting nested metadata values
name = metadata['dimensions']['entities']['values']



#create dictionary for id and country names
name_dict = {entry['id']: entry['name'] for entry in name}


#matching country name from metadata value with main dataset id values
df['Country'] = df['Entity'].map(name_dict)


#filtering dataset to match WHO dataset 2000-2015
filtered_df = df[(df['Year'] >= 2000) & (df['Year'] <= 2015)]
print(filtered_df)

#saving dataset
#filtered_df.to_csv("smoking_deathrate.csv")