import pandas as pd
import json

data="data.json"
metadata="metadata.json"

with open("data.json", encoding='utf-8', errors='ignore') as json_data:
    data = json.load(json_data, strict=False)

with open("metadata.json", encoding='utf-8', errors='ignore') as json_data:
    metadata = json.load(json_data, strict=False)


df = pd.DataFrame({
    'Entity': data['entities'],
    'Year': data['years'],
    'Value': data['values']
})

name = metadata['dimensions']['entities']['values']

name_dict = {entry['id']: entry['name'] for entry in name}

df['Country'] = df['Entity'].map(name_dict)

# Display the DataFrame


filtered_df = df[(df['Year'] >= 2000) & (df['Year'] <= 2015)]
print(filtered_df)

filtered_df.to_csv("smoking_deathrate.csv")