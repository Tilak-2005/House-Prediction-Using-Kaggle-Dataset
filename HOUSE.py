import pandas as pd  
import math
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score,root_mean_squared_error
from sklearn.model_selection import train_test_split

data = pd.read_csv("Housing.csv")
df = pd.DataFrame(data)
# print(df)
df_new = df.copy()

columns = ["mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"]
numerical_cols = ["area","bedrooms","bathrooms","stories"]
columns_with = ["area","bedrooms","bathrooms","stories","mainroad","guestroom","basement","hotwaterheating","airconditioning","parking","prefarea","furnishingstatus"]
scaler = MinMaxScaler()


encoder = OneHotEncoder(drop="first",sparse_output=False)
encoded = encoder.fit_transform(df_new[columns])
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(columns))


scaler = MinMaxScaler()
scaled = scaler.fit_transform(df_new[numerical_cols])
scaled_df = pd.DataFrame(scaled, columns=numerical_cols)


df_final = pd.concat([scaled_df.reset_index(drop=True),
                      encoded_df.reset_index(drop=True),
                      df_new["price"].reset_index(drop=True)], axis=1)

X = df_final.drop("price",axis=1)
y = df_final["price"]

X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,test_size=0.2,random_state=42)
model = LinearRegression()
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test,y_pred)
rmse = root_mean_squared_error(y_test,y_pred)
r2 = r2_score(y_test,y_pred)
for index,i in enumerate(y_pred):
    print(f"The Test Sample Prices of house {index + 1} is  : $ {math.ceil(i)}\n")
print("Mean Squared Error:", round(mse,2))
print("Root Mean Squared Error:", round(rmse,2))
print("R² Score:", round(r2,2))