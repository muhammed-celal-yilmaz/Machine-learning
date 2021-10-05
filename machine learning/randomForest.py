import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
#Farklı algoritmalar tekrardan gerçekleşiyor.Örnek data üzerinden ortalama alınarak tahmin yapılıyor.
data=pd.read_csv("positions.csv")
level=data.iloc[:,1].values.reshape(-1,1)
salary=data.iloc[:,2].values
# y için istemiyor reshape kütüphaneyle alakalı
regression=RandomForestRegressor(n_estimators=100)
#Kaç tane decision tree çalıştıracaksın:
regression.fit(level,salary)
print(regression.predict([[8.3]],random_state=3))
#random state olursa aynı veri çıkar.
#random state=0 ise hep aynı sonuç gelir.