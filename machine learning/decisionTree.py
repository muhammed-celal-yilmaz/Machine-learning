import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
# Decision Tree ile karar yapılarıyla veriyi gruplandırırız.
data=pd.read_csv("positions.csv")

level=data.iloc[:,1:2].values.reshape(-1,1)
salary=data.iloc[:,2].values.reshape(-1,1)

regression=DecisionTreeRegressor()
regression.fit(level,salary)
print(regression.predict([[8.3]]))
# 170k dedi,direk 8 deki adamın fiyatını verdi yani.
#Görselleştirelim:
plt.scatter(level,salary,color="red")
#küçük değerle çiz.
x=np.arange(min(level),max(level),0.01).reshape(-1,1)
plt.plot(x,regression.predict(x),color="orange")
plt.xlabel("level")
plt.ylabel("Salary")
plt.title("Decision Tree Model")
plt.show()
#Merdiven şeklinde grafik oluşturuyor.
