
# Analysis of a nutrition dataset
An analysis of a nutrition dataset in Python. 

This includes:

1) Cleaning the dataset and discovering the mean calorie count for food items containing certain keywords
2) Extracting the mean amount of calories for food entries that included certain keywords
3) Creating a new column "calorie level" that binned calorie counts into levels 1-4 using the minimum, maximum, and quartile values
4) Creating a new column "all vitamins" that added up the total vitamins for each food item.
5) Performing Linear Regression on fat vs. calories (0.65 r^2 score)
6) Performing Linear Regression on water vs. calories (-0.81 r^2 score)
7) Performing cross-validation on multiple Sci-Kit Learn Machine Learning models, ultimately selecting RandomForestClassifier to create the model
8) Predicting calorie level from fat, carbohydrates, and protein at ~95% accuracy using RandomForestClassifier.

