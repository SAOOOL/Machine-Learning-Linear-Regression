import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
#requires pandas and scikit-learn, libraries
#optional libraries are seaborn and matplot (to visualize regression lines)

#pass our csv file into "countries"
countries = pd.read_csv("countries.csv")
countries = countries[[ "country", "year", "athletes", "age", "prev_medals", "medals"]]
countries = countries.dropna()
#find the highest correlations to "medals" column
numeric_cols = countries.select_dtypes(include=['int', 'float']).columns
corr_matrix = countries[numeric_cols].corr()["medals"]
#print(corr_matrix)

#visual representation of regression line of athletes and medals (# prev_medals = .9)
# plot = sns.lmplot(x="prev_medals", y="medals", data=countries, fit_reg=True, ci=None)
# plt.show()

#The test and training sets will be based on a 30/70 split
train_set = countries[countries["year"] < 2008].copy()
test_set = countries[countries["year"] >= 2008].copy()

#load relevant info into linearRegression function from sickit-learn library
model = LinearRegression()
relevant_columns = ["athletes", "prev_medals"]
target = "medals"

#train the model on the training set, then add forecasted medals on the test set
model.fit(train_set[relevant_columns], train_set["medals"]) 
forecast = model.predict(test_set[relevant_columns])
test_set["forecast"] = forecast
#add "forecast" row to test_set normalize the predictions 
test_set.loc[test_set["forecast"] < 0, "forecast"] = 0
test_set["forecast"] = test_set["forecast"].round()
#print(test_set)

#We will be using MAE as the mian error metric since linear regression models cannot use F1 scores and MAE is less sensitive to outliers
#When compared to MSE and RMSE
error = mean_absolute_error(test_set["medals"], test_set["forecast"])
print(f"MAE error Metric for this Model: {error:.2f}")

#logic for input of desired year and country
choice = "n"
valid = False
year = "NaN"
country = "NaN"
print("Please input country name and olympic year(2008-2016)")
while choice == "n":
    while valid == False:
        try: 
            year = int(input("Olympic year: "))
            valid = True
        except Exception:
            valid == False
            print("Invalid input. Please enter a valid year.")
    country = input("Country name: ")
    find_country = test_set[test_set["country"] == country]
    find_year = test_set[test_set["year"] == year]
    if find_country.empty == True:
        print("No data available for", country)
        choice = input("Would you like to exit?: y/n ")
    elif find_year.empty == True:
        print("No data available for", year)
        choice = input("Would you like to exit? y/n ")
    else:
        selected_row = test_set[(test_set['year'] == year) & (test_set['country'] == country)]
        forecast_value = selected_row.loc[:, 'forecast'].values[0]
        forecast_value = int(forecast_value)
        true_value = selected_row.loc[:, 'medals'].values[0]
        true_value = int(true_value)
        print(f"Predicted Olympic Medals for {country} in {year} is {forecast_value}")
        print(f"Meanwhile the actual value was {true_value}")
        choice = "y"