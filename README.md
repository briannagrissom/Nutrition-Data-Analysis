# This is an analysis of a nutrition dataset in python. 
# I started by cleaning the dataset, then seeing which food items had the least amount of calories. 
# I extracted the mean amount of calories for food entries that included certain keywords.
# I created a new column "calorie level" that created bins of calorie counts into levels 1-4 using the minimum, maximum, and quartile values for calories.
# I created a new column "all vitamins" that added up total vitamins for each food item.
# I performed Linear Regression on fat vs. calories, finding a 0.65 r^2 score between these variables.
# I performed Linear Regression on water vs. calories, finding a -0.81 r^2 score between these variables.
# I perfomed cross-validation on multiple Sci-Kit Learn ML models, choosing to use RandomForestClassifier.
# I predicted calorie level from fat, carbohydrates, and protein at ~95% accuracy using RandomForestClassifier
