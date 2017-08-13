def model(weight, data):
    """
    Model function definition.

    Currently model setting:
        y = sum(b + weight[i] * data[i])
        n = dim(weight)
        b = weight[0]
        i = 1, ..., n-1

    This function should be overwrite if your need different model.

    Args:
        weight:
            A list of floats, a vector of weights.
        data:
            A list of floats, a vector of data.

    Returns:
        y:
            A float, model function output value.
    """

    # initialize with bias
    y = weight[0]

    # calculate sum of data mutiplied by weight
    for i in range(1,len(weight)):
        y = y + weight[i] * data[i-1]

    return y


def loss_func_root(weight, que_data, ans_data):
    """
    Loss function without square.

    Useful for calculating gradient of function.
    If you need the true loss function, just do a square of its return.

    Args:
        weight:
            A list of floats, weights of model function.
        que_data:
            A list of floats as inputs of model function.
        ans_data:
            A list of floats, target function real outputs.

    Returns:
        err_rt:
            A float, model function estimation error without square.
    """

    # calculate estimation error without square
    err_rt = ans_data - model(weight, que_data)

    return err_rt

def loss_func(weight, que_data_sets, ans_data_sets):
    """
    Loss function for calculating estimation error.

    Formula:
        L = sum((ans_data_sets[i] - model(weight, que_data_sets[i]))^2)
        n = dim(que_data_sets)
        i = 1, ..., n
    
    Args:
        weight:
            A list of floats, weights of model function.
        que_data:
            A list of floats as inputs of model function.
        ans_data:
            A list of floats, target function real outputs.

    Returns:
        err:
            A float, model function estimation error.
    """

    # calculate estimation error
    err = sum([
                pow(loss_func_root(weight, que, ans), 2) 
                for (que, ans) in zip(que_data_sets, ans_data_sets)
             ])

    return err


def diff_func_set(que_data_sets, ans_data_sets):
    """
    Return gradient of loss function.

    Args:
        weight:
            A list of floats, weights of model function.
        que_data:
            A list of floats as inputs of model function.
        ans_data:
            A list of floats, target function real outputs.

    Returns:
        gradient:
            A list of function, gradient of loss function.
    """

    # initialize with partial of bias
    gradient = [lambda w: 
                    sum([
                        -2 * loss_func_root(w, que, ans)
                        for (que, ans) in zip(que_data_sets, ans_data_sets)
                    ])
               ]
   
    # get paritial derivation of each weight
    for index in range(1,len(que_data_sets[0])+1):
        gradient.append(
                        lambda w: 
                            sum([
                                -2 * loss_func_root(w, que, ans) * w[index]
                                for (que, ans) in zip(que_data_sets, ans_data_sets)
                            ])
                       )

    return gradient
    

def rate_func(iter_time):
    """
    Learning rate for gradient descent.

    Args:
        iter_time:
            An integer, Currently iterating time.

    Returns:
        n:
            A float constant.
    """

    n = 0.000000001

    return n

