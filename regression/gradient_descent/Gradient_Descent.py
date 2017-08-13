class Gradient_Descent:
    """
    Do Gradient Descent and return weight vector (list).

    This class is used to calculate Gradient Descent for a function.
    Users will provide a function's differential to find local minimum.
    It also serves as a base class for variant of Gradient Descent.

    Attributes:
        diff_func:
            A list of functions, a vector of target function's differential.
        dim:
            An integer, dimension of diff_func.
        rate_func: 
            A function, learning rate of Gradient Descent.
    """

    def __new__(cls,
                diff_func = None,
                rate_func = None):
        """
        Create instance if arguments are correctly pass.
        
        Args:
            diff_func:
                A list of functions, a vector of target function's differential.
            dim:
                An integer, dimension of diff_func.
            rate_func: 
                A function, learning rate of Gradient Descent.

        Raises:
            RuntimeError:
                If one of the arguments is not set.
        """

        if not diff_func:
            raise RuntimeError('diff_func is not set.')
        if not rate_func:
            raise RuntimeError('rate_func is not set.')

        return super(Gradient_Descent, cls).__new__(cls)


    def __init__(self,
                 diff_func = None,
                 rate_func = None):
        """
        Set attributes for later usage.

        Args:
            diff_func:
                A list of functions, a vector of target function's differential.
            dim:
                An integer, dimension of diff_func.
            rate_func: 
                A function, learning rate of Gradient Descent.
        """

        self.diff_func = diff_func
        self.dim = len(diff_func) # dimension of vector diff_func
        self.rate_func = rate_func

    def formula(self, w_in, iter_time):
        """
        Formula to calculate Gradient Descent.

        Current Formula:
            w<t> = w<t-1> - rate_func(t-1) * diff_func(w<t-1>)

        This method should be overrided when you inherit this class
        to create your own version of Gradient Descent.

        Args:
            w_in: 
                A list of floats, a vector as a input of target function.
            iter_time:
                An integer, currently iterating round.

        Returns:
            w_out: 
                A list of floats, Gradient Descent output vector.
        """

        w_out = []
        
        #testcodesection
        #print('iter_time: %d' % iter_time)
        #print('input vector: ' + w_in.__str__())
        #testcodesection

        for index in range(0, self.dim):
            w_out.insert(index, 
                        w_in[index] - self.rate_func(iter_time) * self.diff_func[index](w_in))

        return w_out
        

    def run(self,
            start = None,
            iter_time = 0):
        """
        Run Gradient Descent iter_time times with start vector start.

        Args:
            start: 
                A list of floats, a vector as a start point.
            iter_time: 
                An integer, iterating times for Gradient Descent.

        Returns:
            w_out: 
                A list of floats, output vector for local minimum.

        Raise:
            RuntimeError: 
                If start not set or iter_time equals to 0.
        """

        if not start:
            raise RuntimeError('start is not set.')
        if not iter_time:
            raise RuntimeError('iter_time is not set.')
            
        w_in = start

        for t in range (0, iter_time):
            w_out = self.formula(w_in, t)
            w_in = w_out

        return w_out

    













    
