#!/usr/bin/env python
# coding: utf-8

# In[474]:


import pandas as pd

df = pd.read_excel('/Users/briannagrissom/Downloads/nutrition2.xlsx.xlsx') # load dataset

old_df = df.copy()

df = df.drop(columns = 'vitamin_a_IU') # drop column


# In[475]:


df.drop(columns = df.columns[0], inplace = True) # remove unnamed column

df.head()


# In[238]:


df.shape # the dataframe has 8789 rows, 22 columns


# In[843]:


df.serving_size_grams.unique() # serving sizes are all 100g 

# Do meals with certain food groups have more calories than others?

df_calories = df[['name','calories']] # create new data frame that contains only the food name and calories

print(df_calories.loc[df_calories.calories== df_calories.calories.max(),'name'])
# fish with fish oil and Lard are the most calorie dense, at 902 calories

df_calories.loc[df_calories.calories== df_calories.calories.min(),'name'] 
# beverages, sweeteners, and salt are the least calorie dense, at 0 calories

meats_calories = df_calories.loc[df_calories.name.str.contains('meat'),'calories'] # create new DF that has only meats
meats_calories.mean() # mean amount of calories for foods that contains 'meat': 193

df_calories[df_calories.name.str.contains('cheese')]['calories'].mean()
# mean amount of calories for foods that contain "cheese": 271 

df_calories[df_calories.name.str.contains('fruit')]['calories'].mean() 
# mean amount of calories for foods that contain "fruit": 122

df_calories[df_calories.name.str.contains('vegetables')]['calories'].mean() 
# mean amount of calories for foods that contain "vegetables": 117

df_calories[df_calories.name.str.contains('bread')]['calories'].mean()
# mean amount of calories for foods that contain "bread": 284


# In[944]:


mean = df_calories.calories.mean() # mean number of calories for all foods in the data frame: 226 

minimum = df_calories.calories.min() # minimum number of calories: 0

maximum = df_calories.calories.max() # maximum number of calories: 902

# break each food into calorie levels:

q1 = df_calories.calories.describe()[4] # 25th percentile : 91 calories

q2 = df_calories.calories.describe()[5] # 50th percentile: 191 calories

q3 = df_calories.calories.describe()[6] # 75th percentile: 337 calories

df_calories['calorie level'] = pd.cut(df_calories.calories, bins = [minimum-0.01, q1, q2, q3, maximum], labels = [1, 2, 3, 4])
df['calorie level'] = pd.cut(df_calories.calories, bins = [minimum-0.01, q1, q2, q3, maximum], labels = [1, 2, 3, 4] )
# create levels for the calories based on the quartiles calculated above

df_calories.loc[df_calories['calorie level'] == 2,:][100:105] # print a few rows where calorie level = 2



# In[876]:


df.calories.value_counts()[0] # most frequently occuring calorie count for a food item is 884 calories, at 78 entries

df['all_vitamins'] = df.vitamin_b12_mcg + df.vitamin_b6_mg + df.vitamin_c_mg + df.vitamin_d_IU + df.vitamin_e_mg + df.vitamin_k_mcg
# create new column that adds up the vitamins for each food item

df.loc[df.all_vitamins == df.all_vitamins.max(),:] 
# "fish oil, cod liver" has the most total vitamins, at 10000 IU of vitamin D and 0 of all the other vitamins

df.sort_values('all_vitamins', ascending = False)[:5] # the top 5 foods with the highest amount of vitamins

dfnew = df.drop(columns='name')
stats_by_level = dfnew.groupby('calorie level').agg(['sum','mean']) 
# find the mean and sum of each nutrient by the calorie level

stats_by_level

# some statistics by calorie level:

# calorie level 4 had the highest mean & sum for fat, sodium, carbs, fiber, sugar, all vitamins, calcium, iron
# calorie level 1 had the highest mean & sum for water


# In[924]:


import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
# import modules


# What is the association between fat and calories? 

x = np.array(df.total_fat_grams).reshape(-1,1) 
# create inputs: an array of the total fat

y = np.array(df.calories) 
# create outputs: an array of the calorie amount

linearmodel = LinearRegression().fit(x, y) # create the linear regression model

predicted_yvals = list(map(lambda n: linearmodel.intercept_ + (linearmodel.coef_ * n), x))
# calculate the predicted calorie amounts for every given fat count

reshaped_yvals = np.array(predicted_yvals).reshape(-1,1) 
# reshape predicted values into a 2D array
plt.figure(figsize=(20,6))
plt.scatter(x,y, s=1.5)
plt.plot(x, reshaped_yvals)
plt.xlabel('Fat (g)')
plt.ylabel('Calories') # clear positive association between fat and calories
plt.title('Fat vs. Calories')
plt.show()
# plot fat vs. calories and the linear rgression line

linearmodel.predict([[20]]) 
# for a serving that has 20g of fat, it's predicted to have 308 calories

r2_score(y, reshaped_yvals)
# 0.65 accuracy score between the actual calorie values and the predicted calorie values


# In[851]:


df[df.name.str.contains('yogurt')].describe() 
# foods that contain "yogurt" have a mean of 160 calories, mean of 6.38g fat

df[df.name.str.contains('egg')].describe() 
# foods that contain "egg" have a mean of 211 calories, mean of 8.27g fat 

#consistent with regression in that foods with more calories tend to have more fat


# In[923]:


import matplotlib.pyplot as plt 
import numpy as np
from sklearn.linear_model import LinearRegression

# how are calories and water (g) related?

x = np.array(df.water_grams).reshape(-1,1)
# set input as an array of the amounts of water

y = np.array(df.calories)
# set output as an array of the amount of calories

linearmodel = LinearRegression().fit(x,y)
# create model

yvals = list(map(lambda n: linearmodel.intercept_ + (linearmodel.coef_ * n), x))
# create a list of the predicted calories

new_yvals = np.array(yvals).reshape(-1,1)
# reshape the list to be 2 dimensional
plt.figure(figsize=(20,6))
plt.scatter(x,y, s=8)
plt.plot(x, new_yvals, lw=3, color= 'navy')
plt.xlabel('Water (g)')
plt.ylabel('Calories')
plt.title('Water vs. Calories')
plt.show()
# plot linear regression line and a scatter plot of water vs. calories

linearmodel.predict([[40]]) 
# for a food item with 40g of water, the model predicts it will have 291 calories

linearmodel.score(x,y)
# R^2 of -0.81: negative relationship between water and calories




# predict the calorie level from the 3 main macronutrients: fat, protein, and carbohydrates
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
# import modules


df2 = df[['name','total_fat_grams','protein_grams','carbohydrate_grams', 'calorie level']]
# create a new data frame with the desired columns and 600 entries
print(f'Create a new data frame with the desired columns (600 entries): \n {df2.head(2)}\n')

features = df2.drop(columns=['name','calorie level'])
# set the features by removing columns
print(f'Create input data frame by dropping off the colums "name" and "calorie level:\n {features.head(2)}\n')

features2 = StandardScaler().fit_transform(features)
# scale the features to improve the model

output = df2['calorie level']
# set output as the calorie level

print(f'Create data frame of the output, the calorie level:\n{output.head(2)}\n')
print()
print('Apply RandomForestClassifier to try to predict the calorie level from a given amount of each input\n')

X_train, X_test, y_train, y_test = train_test_split(features2, output, test_size=0.3)
# set 70% of the data for training the model and the remaining 30% for testing the model
RFC = RandomForestClassifier(n_estimators=70, random_state=100)
# create the model
print(f'The model:\n{RFC}\n')

RFC.fit(X_train, y_train)
# fit the model to the training data

score = RFC.score(X_test, y_test)
# find the accuracy score for the model
print(f'The accuracy score for this model is {score}\n')


y_predicted = RFC.predict(X_test)
# assign the predicted calorie levels to a variable

CM = confusion_matrix(y_test,y_predicted)
plottedCM = ConfusionMatrixDisplay(CM, display_labels=[1,2,3,4])
# create a confusion matrix
print(f'The confusion matrix for this model is \n')
plottedCM.plot()
plt.show()

importance = RFC.feature_importances_
print(f'The feature importance for fat: {importance[0]}\n')
print(f'The feature importance for protein: {importance[1]}\n')
print(f'The feature importance for carbohydrates: {importance[2]}\n')
# find the feature importances for each macronutrient

print()
print('_______________________________________________________')
print()
print()

df2['predicted calorie level'] = RFC.predict(features2) 
# let's see how accurate the model was

wrong = df2.loc[df2['calorie level'] != df2['predicted calorie level'],:]
# filter rows where the calorie level is not equal to the predicted calorie level

print(f'A few rows where the model predicted incorrectly:\n {wrong.sample(5)}')

print(f'How many rows the model predicted incorrectly: {len(wrong)}\n')
print()
right = df2.loc[df2['calorie level'] == df2['predicted calorie level'],:]
#filter rows where the calorie level is equal to the predicted calorie level

print(f'A few rows where the model predicted correctly:\n {right.sample(5)}')
print()
print(f'How many rows the model predicted correctly: {len(right)}')
print()
print(f'Success rate: {len(right)/len(df2)}')


# In[961]:


from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier()
params = {'n_estimators':[10,20,30,40,50,60,70,80,90,100,110,120,130,140,150],
         'criterion' : ["gini", "entropy"]}
best = GridSearchCV(rfc, params)
best.fit(features2, output)
print(best.best_params_)
print(best.best_score_)


# In[ ]:




