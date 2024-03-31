import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import Dense
import pandas as pd
import json

df = pd.read_csv("UnstopData.csv")

print(df)

df2 = df[['easySolved','medSolved','diffSolved','enagementHrs','enagementMins']]
df3 = df[['UnstoppableGuaranteeScore']]


#Scaling the dataframe as per required weightage

weights = np.array([0.7, 0.85, 0.9, 0.4, 0.3])

scaledf = df2.mul(weights, axis=1)
print(scaledf)

X_weighted = scaledf.iloc[:300]

y = df3.iloc[:300]

# Define the neural network architecture
model = Sequential()
model.add(Dense(8, input_dim=5, activation='relu'))  
model.add(Dense(6, activation='relu'))               # Hidden layer
model.add(Dense(1, activation='linear'))             # Output layer

model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(X_weighted, y, epochs=90, batch_size=2)

score_predictor = model.to_json()
with open("score_predictor.json", "w") as jf:
    jf.write(score_predictor)

model.save("score_predictor.h5")


