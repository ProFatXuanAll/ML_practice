from Gradient_Descent import Gradient_Descent

"""
answer function:
    y=x1+x2+x3

question data set:
    [(1,1,1), (1,2,3), (-4,7,9)]

answer data set:
    [3,6,12]

model:
    y' = x0+x1*d1+x2*d2+x3*d3 (linear regression model function)
"""

def diff_func0(x):
    # b+w1*x1+w2*x2+w3*x3
    return -2*(3-(x[0]+x[1]+x[2]+x[3])+6-(x[0]+x[1]+2*x[2]+3*x[3])+12-(x[0]+-4*x[1]+7*x[2]+9*x[3]))

def diff_func1(x):
    return -2*((3-(x[0]+x[1]+x[2]+x[3]))*1+(6-(x[0]+x[1]+2*x[2]+3*x[3]))*1+(12-(x[0]+-4*x[1]+7*x[2]+9*x[3]))*(-4))

def diff_func2(x):
    return -2*((3-(x[0]+x[1]+x[2]+x[3]))*1+(6-(x[0]+x[1]+2*x[2]+3*x[3]))*2+(12-(x[0]+-4*x[1]+7*x[2]+9*x[3]))*7)

def diff_func3(x):
    return -2*((3-(x[0]+x[1]+x[2]+x[3]))*1+(6-(x[0]+x[1]+2*x[2]+3*x[3]))*3+(12-(x[0]+-4*x[1]+7*x[2]+9*x[3]))*9)


def rate_func(t):
    return 0.001

diff_func_vec=[diff_func0,diff_func1,diff_func2,diff_func3]
gd = Gradient_Descent(diff_func_vec, rate_func)
start = [1,1,1,1]
iter_time = 10000

print(gd.run(start,iter_time))

