import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)
df=pd.read_csv(r"C:\Users\HI\Downloads\IRIS.csv")
df_cpy=df.copy()
print(df.head())
df=df.drop_duplicates(keep="last")
df=df.reset_index(drop=True)
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df["species"]=le.fit_transform(df["species"])
IndepVar = []
for col in df.columns:
    if col != 'species':
        IndepVar.append(col)

TargetVar = 'species'

x = df[IndepVar]
y = df[TargetVar]
from sklearn.model_selection import train_test_split 

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.30, random_state=42)
from sklearn.ensemble import RandomForestClassifier
model = RandomForestClassifier()
model.fit(x_train,y_train)
y_pred = model.predict(x_test)
import pickle
pickle.dump(model,open("E:\ML Projects\iris1.pkl",'wb'))



