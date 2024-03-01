import pandas as pd
from pandas_profiling import ProfileReport

df = pd.read_csv('admissio.csv')

profile = ProfileReport(df)#(df, minimul= True)
profile.to_file(output_file="admisfor.html")