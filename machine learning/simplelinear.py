import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
data=pd.read_csv("hw_25000.csv")

# print(data.columns)
#Görüntüyü görselleştirmek için:
# x-y diyagramında:line fit, MSE
#25000 data grafikleşir:
# plt.scatter(data.Height,data.Weight)
# plt.xlabel("Boy")
# plt.ylabel("Kilo")
# plt.show()


#Şimdi tahminleme yapalım yani ortalama çizgi çekip,değeri bulalım.
#25000,1 formatında değerleri alalım.
boy=data.Height.values.reshape(-1,1)
kilo=data.Weight.values.reshape(-1,1)
#Line fit ediyoruz böylece,tahmin değeri
regression=LinearRegression()
regression.fit(boy,kilo)
#bu regresyon modeli için tahmin yapalım
print(regression.predict([[70]]))
#Bu şekilde bize boy olarak 133.267 döndü.

#Tahmin değerlerini görselleştirelim:
plt.scatter(data.Height,data.Weight)
#Burda x için min ve max değerlerini alıyoruz
x=np.arange(min(data.Height),max(data.Height)).reshape(-1,1)
#Burda x e göre y değerlerini bukuyoruz ve hayali çizgi çekiliyor.
plt.plot(x,regression.predict(x),color="red")
#Ve grafiğimiz line fit ile çizilecek.
plt.xlabel("Boy")
plt.ylabel("Kilo")
plt.title("Simple Linear Regression Model")
plt.show()

# Kullandığımız algoritmanın R-SQUARE yöntemiyle doğruluğunu bulalım.
# Bu doğruluk 1'e yakınlığı ölçüsünde değer kazanır.
print(r2_score(kilo,regression.predict(boy)))
#0,25 verdi ama boy ve kilo bilindiği üzere çok değişkendir.















