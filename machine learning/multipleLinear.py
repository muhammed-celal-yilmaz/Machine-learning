import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
#birden fazla değerle çalışmak 
data=pd.read_csv("insurance.csv")
print(data.columns)
#yaş ve bmi değerine göre harcamayı bulmak mesela.
#expenses yani y değeri bulmak istiyoruz.
# y ekseni
expenses=data.expenses.values.reshape(-1,1) #formata uygun olsun diye.
# x ekseni
#Değerleri x eksenine alcaz.0 ve 2 elemanını al.
ageBmis=data.iloc[:,[0,2]].values # 0 ve 1.sütunları alıyoruz.
regression=LinearRegression()
regression.fit(ageBmis,expenses)
print(regression.predict(np.array([[20,20]])))
#Harcama 5000 çıktı mesela.
print(regression.predict(np.array([[20,20],[20,25],[20,26]])))
#Yaş sabit tutulunca beden kitle endeksinin artması harcamarı doğru oranda etkiliyor..
#Sistemi böylece eğittik.
#Yanılma payımız yüksek değil burda