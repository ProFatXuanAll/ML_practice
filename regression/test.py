import get_train_data
import get_test_data
import linear_regression
import gradient_descent

import random

# get training data
que_data_sets, ans_data_sets = get_train_data.path('data/train.csv')

# start position vector
start = []

# randomly pick start point
for i in range(0,18):
    start.append(random.random())

# setup linear regression over gradient descent
gd = gradient_descent.Gradient_Descent(
        linear_regression.diff_func_set(
            que_data_sets,
            ans_data_sets
        ),
        linear_regression.rate_func
    )

# run gradient descent
res = gd.run(start,1000)

# get testing data
que_data_sets, ans_data_sets = get_test_data.path('data/test.csv')

print('loss:')
print(linear_regression.loss_func(
        res,
        que_data_sets,
        ans_data_sets)
     )
