import pandas as pd
import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

# load dataset
df = pd.read_excel('/Users/briannagrissom/Downloads/nutrition2.xlsx.xlsx') 

old_df = df.copy()

# drop column
df = df.drop(columns = 'vitamin_a_IU') 

# remove unnamed column
df.drop(columns = df.columns[0], inplace = True) 

# serving sizes are all 100g 
print(df.serving_size_grams.unique()) 

# Do meals with certain food groups have more calories than others?

df_calories = df[['name','calories']] # create new data frame that contains only the food name and calories

# "fish with fish oil" and Lard are the most calorie dense, at 902 calories
print(df_calories.loc[df_calories.calories== df_calories.calories.max(),'name'])

# beverages, sweeteners, and salt are the least calorie dense, at 0 calories
print(df_calories.loc[df_calories.calories== df_calories.calories.min(),'name'])

# mean amount of calories for foods that contains "meat": 193
print(df_calories[df_calories.name.str.contains('meat')]['calories'].mean())

# mean amount of calories for foods that contain "cheese": 271 
print(df_calories[df_calories.name.str.contains('cheese')]['calories'].mean())

# mean amount of calories for foods that contain "fruit": 122
print(df_calories[df_calories.name.str.contains('fruit')]['calories'].mean())

# mean amount of calories for foods that contain "vegetables": 117
print(df_calories[df_calories.name.str.contains('vegetables')]['calories'].mean())

# mean amount of calories for foods that contain "bread": 284
print(df_calories[df_calories.name.str.contains('bread')]['calories'].mean())

# mean number of calories for all foods in the data frame: 226 
print(df_calories.calories.mean())

# minimum number of calories: 0
print(df_calories.calories.min())

# maximum number of calories: 902
print(df_calories.calories.max())

# break each food into calorie levels:

# 25th percentile : 91 calories
q1 = df_calories.calories.describe()[4]

# 50th percentile: 191 calories
q2 = df_calories.calories.describe()[5]

# 75th percentile: 337 calories
q3 = df_calories.calories.describe()[6] 

# create levels for the calories based on the quartiles calculated above
df_calories['calorie level'] = pd.cut(df_calories.calories, bins = [minimum-0.01, q1, q2, q3, maximum], labels = [1, 2, 3, 4])
df['calorie level'] = pd.cut(df_calories.calories, bins = [minimum-0.01, q1, q2, q3, maximum], labels = [1, 2, 3, 4] )

# print a few rows where calorie level = 2
print(df_calories.loc[df_calories['calorie level'] == 2,:][100:105])

# most frequently occuring calorie count for a food item is 884 calories, at 78 entries
print(df.calories.value_counts()[0])

# create new column that adds up the vitamins for each food item
df['all_vitamins'] = df.vitamin_b12_mcg + df.vitamin_b6_mg + df.vitamin_c_mg + df.vitamin_d_IU + df.vitamin_e_mg + df.vitamin_k_mcg

# "fish oil, cod liver" has the most total vitamins, at 10000 IU of vitamin D and 0 of all the other vitamins
df.loc[df.all_vitamins == df.all_vitamins.max(),:] 

# the top 5 foods with the highest amount of vitamins
df.sort_values('all_vitamins', ascending = False)[:5]

# find the mean and sum of each nutrient by the calorie level
dfnew = df.drop(columns='name')
stats_by_level = dfnew.groupby('calorie level').agg(['sum','mean']) 
print(stats_by_level)

# some statistics by calorie level:

# calorie level 4 had the highest mean & sum for fat, sodium, carbs, fiber, sugar, all vitamins, calcium, iron
# calorie level 1 had the highest mean & sum for water


# What is the association between fat and calories? 

# create inputs: an array of the total fat
x = np.array(df.total_fat_grams).reshape(-1,1) 

# create outputs: an array of the calorie amount
y = np.array(df.calories) 

# create the linear regression model
linearmodel = LinearRegression().fit(x, y) 

# calculate the predicted calorie amounts for every given fat count
predicted_yvals = list(map(lambda n: linearmodel.intercept_ + (linearmodel.coef_ * n), x))

# reshape predicted values into a 2D array
reshaped_yvals = np.array(predicted_yvals).reshape(-1,1) 

# plot the data with the regression line. Clear positive association between fat and calories.
plt.figure(figsize=(20,6))
plt.scatter(x,y, s=1.5)
plt.plot(x, reshaped_yvals)
plt.xlabel('Fat (g)')
plt.ylabel('Calories'
plt.title('Fat vs. Calories')
plt.show()

# for a serving that has 20g of fat, it's predicted to have 308 calories
print(linearmodel.predict([[20]]))

# 0.65 accuracy score between the actual calorie values and the predicted calorie values
print(r2_score(y, reshaped_yvals))

# foods that contain "yogurt" have a mean of 160 calories, mean of 6.38g fat
print(df[df.name.str.contains('yogurt')].describe())

# foods that contain "egg" have a mean of 211 calories, mean of 8.27g fat 
print(df[df.name.str.contains('egg')].describe())


# this is consistent with regression in that foods with more calories tend to have more fat


# how are calories and water (g) related?

# set input as an array of the amounts of water
x = np.array(df.water_grams).reshape(-1,1)

# set output as an array of the amount of calories
y = np.array(df.calories)

# create model
linearmodel = LinearRegression().fit(x,y)

# create a list of the predicted calories
yvals = list(map(lambda n: linearmodel.intercept_ + (linearmodel.coef_ * n), x))

# reshape the list to be 2 dimensional
new_yvals = np.array(yvals).reshape(-1,1)

# plot linear regression line and a scatter plot of water vs. calories
plt.figure(figsize=(20,6))
plt.scatter(x,y, s=8)
plt.plot(x, new_yvals, lw=3, color= 'navy')
plt.xlabel('Water (g)')
plt.ylabel('Calories')
plt.title('Water vs. Calories')
plt.show()

# for a food item with 40g of water, the model predicts it will have 291 calories
print(linearmodel.predict([[40]]))

# R^2 of -0.81: negative relationship between water and calories
print(linearmodel.score(x,y))


# predict the calorie level from the 3 main macronutrients: fat, protein, and carbohydrates

# create a new data frame with the desired columns and 600 entries
df2 = df[['name','total_fat_grams','protein_grams','carbohydrate_grams', 'calorie level']]
print(f'Create a new data frame with the desired columns (600 entries): \n {df2.head(2)}\n')

# set the features by removing columns
features = df2.drop(columns=['name','calorie level'])
print(f'Create input data frame by dropping off the colums "name" and "calorie level:\n {features.head(2)}\n')

# scale the features to improve the model
features2 = StandardScaler().fit_transform(features)

# set output as the calorie level
output = df2['calorie level']
print(f'Create data frame of the output, the calorie level:\n{output.head(2)}\n')
print()
print('Apply RandomForestClassifier to try to predict the calorie level from a given amount of each input\n')

# set 70% of the data for training the model and the remaining 30% for testing the model
X_train, X_test, y_train, y_test = train_test_split(features2, output, test_size=0.3)
# create the model
RFC = RandomForestClassifier(n_estimators=70, random_state=100)

# fit the model to the training data
RFC.fit(X_train, y_train)

# find the accuracy score for the model
score = RFC.score(X_test, y_test)
print(f'The accuracy score for this model is {score}\n')

# assign the predicted calorie levels to a variable
y_predicted = RFC.predict(X_test)

# create a confusion matrix
CM = confusion_matrix(y_test,y_predicted)
plottedCM = ConfusionMatrixDisplay(CM, display_labels=[1,2,3,4])
print(f'The confusion matrix for this model is \n')
plottedCM.plot()
plt.show()

# find the feature importances for each macronutrient
importance = RFC.feature_importances_
print(f'The feature importance for fat: {importance[0]}\n')
print(f'The feature importance for protein: {importance[1]}\n')
print(f'The feature importance for carbohydrates: {importance[2]}\n')


print()
print('_______________________________________________________')
print()
print()

# let's see how accurate the model was
df2['predicted calorie level'] = RFC.predict(features2) 

# filter rows where the calorie level is not equal to the predicted calorie level
wrong = df2.loc[df2['calorie level'] != df2['predicted calorie level'],:]
print(f'A few rows where the model predicted incorrectly:\n {wrong.sample(5)}')
print(f'How many rows the model predicted incorrectly: {len(wrong)}\n')
print()

#filter rows where the calorie level is equal to the predicted calorie level
right = df2.loc[df2['calorie level'] == df2['predicted calorie level'],:]
print(f'A few rows where the model predicted correctly:\n {right.sample(5)}')
print()
print(f'How many rows the model predicted correctly: {len(right)}')
print()
print(f'Success rate: {len(right)/len(df2)}')

# Find the best parameters to predict the calorie level from fat, protein, and carbohydrates
rfc = RandomForestClassifier()
params = {'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150],
         'criterion' : ["gini", "entropy"]}
best = GridSearchCV(rfc, params)
best.fit(features2, output)
print(best.best_params_)
print(best.best_score_)






