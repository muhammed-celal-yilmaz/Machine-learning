import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
data=pd.read_csv("positions.csv")
print(data.columns)
#Linear regression:Linear düzlemde gösterelim.
#Seviyeye göre maaş:
level=data.iloc[:,1].values.reshape(-1,1)
salary=data.iloc[:,2].values.reshape(-1,1)
regression=LinearRegression()
regression.fit(level,salary)
tahmin=regression.predict([[8.3]])
#239k dedi halbuki arada 90k var ve bu ciddi artış yani doğru model değil.
plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.show()
#Evet artış polinomal
#Bu seviye bu arada
#Yani linear yöntemle bulamayız.
#Her level için değerlere göre çizgi çizelim:
print(r2_score(level,regression.predict(level)))
#Bu da hata payı


#Burda ise yanılma payımızı daha profesyonelce hazırlamalıyız.
#Doğrusal bir durum yok yani.
#MSE için en uygun dereceyide ver.
regressionPoly=PolynomialFeatures(degree=4)
levelPoly=regressionPoly.fit_transform(level) #♦level polinomal dizayn edildi.
regression2=LinearRegression()
#Ama polinomal yaptık.
regression2.fit(levelPoly,salary)
#Yani 8.3 için de polinom değerini fit edelim.
tahmin2=regression2.predict(regressionPoly.fit_transform([[8.3]]))
#189k gibi daha doğru bir değer döndü.
plt.scatter(level,salary,color="red")
plt.plot(level,regression.predict(level),color="blue")
plt.plot(level,regression2.predict(levelPoly))
plt.show()
#Yapımızı daha düzgün çıkardı grafikten görürsek.








