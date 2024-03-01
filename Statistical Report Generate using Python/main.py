import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('data.csv')
print(df)

profile = ProfileReport(df)#(df, minimul= True)
profile.to_file(output_file="realestate.html")