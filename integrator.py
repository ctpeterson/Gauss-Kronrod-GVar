""" External modules """
import numpy as np # For number crunching
import gvar as gv # For error propagation
import functools as functools # For implementing recursion

"""Define class to implement n-dimensional integration

Description:
-------

Defines class that implements n-dimensional integration
by recursively calling one-dimensional Gauss-Kronrod
adaptive integrator. The adaptive integrator itself
creates the adaptive grid with recursive interval
splitting over an 21-node Gauss-Kronrod quadrature.
"""
class NDimAdapGK(object):
    """Class implementing n-dimensional numerical integration


    Description:
    -------

    This class implements n-dimensional numerical integration
    by recursively calling a one-dimensional adaptive
    Gauss-Kronrod integrator.


    Example:
    -------

    Integrate over f(x, y, z) = x^2 * y^2 * z^2 with x in [1, 3],
    y in [2, 4] and z in [8, 10]

    >> f = lambda x, y, z : x**2. * y**2. * z**2.
    >> ranges = [[.1, 3.], [2., 4.], [8., 10.]]
    >> integrator = NDimAdapGK()
    >> res, err = integrator(f, ranges)


    Attributes:
    -------
       gk_args (dict): Arguments going into gk integrator
       abscis (array): Abscissa for GK
       lw_prs_wghts (array): Low-precision integral weights
       hg_prs_wghts (array): High-precision integral weights
       f (func): Function to be integrated over
       ranges (array): Array of ranges fed into gk integrator
       max_depth (int): Number of dimensions of integral
    """
    def __init__(self, args = (), tol = 1e-5):
        """Initialize integrator object


        Description:
        -------

        Initializes integration object. Defines parameters to be put into
        Gauss-Kronrod integrator.


        Attributes:
        -------
           args (tuple): Tuple of function argumnets
           eps (float): Tolerance for interval splitting
        """
        # Set arguments to gk integrator
        self._gk_args = {'args' : args, 'tol' : tol}

        # Set abscissa and weights
        self._get_gk_abs_wghts()

        # Return nothing
        return None

    def __call__(self, f, ranges):
        """Calculate integral of f over ranges


        Description:
        -------

        Sets up integration and calls integration method
        to calculate n-dimensional integral.

        Not currently capable of dealing with infinite ranges.

        Attributes:
        -------
           f (func): Input function to be integrated
           ranges (array): List of lists containing integration ranges
           y (float): Value of integral
           err (float): Error in outer integral
        """

        """ Set a few things up """
        # Set function to integrate over
        self.f = np.vectorize(f)

        # Set ranges
        self.ranges = ranges

        # Set maximum depth
        self.max_depth = len(self.ranges)

        """ Perform some checks """
        # Cycle through ranges
        for range_ind, range in enumerate(self.ranges):
            # Check to make sure nothing is infinite
            if any(not np.isfinite(bound) for bound in range):
                # Warn user
                print('Warning! Found infinite bound in interval',
                      str(range_ind + 1) + '.',
                      '\nNot performing integration.')

                # Return nothing
                return None

        """ Perform integration """
        # Call n-dimensional adaptive Gauss-Kronrod code
        y = self._adap_gk_nd()

        # Return value of integral and error
        return y

    def _gk(self, f, a, b):
        """Gauss-Kronrod integration


        Description
        -------

        Gauss-Kronrod integration. Calculates one-dimensional integrals.
        Based on Geotecha's implementation of Gauss-Kronrod integration.


        References
        --------
        [1] https://pythonhosted.org/geotecha/_modules/geotecha/mathematics/
            quadrature.html#gk_quad


        Attributes:
        -------
           f (func): Function to be integrated
           a (float): Lower boundary
           b (float): Upper boundary
           mid_diff (float): Middle of difference between boundaries
           mid_int (float): Middle of integration intervale
           func_eval (float): Value of function at evaluated points
           low_pres_intg (float): Low-precision integral
           high_pres_intg (float): High-precision integral
           err (float): Difference between high-pres. and low-pres. integs.
        """
        # Get midpoint of difference
        mid_diff = (b - a) / 2.

        # Get midpoint of interval
        mid_int = (b + a) / 2.

        # Rescale abscissae
        abcis_resc = mid_diff * self.abscis + mid_int

        # Evaluate function
        func_eval = f(abcis_resc)

        # Get lower precision integral
        low_pres_intg = mid_diff * np.sum(func_eval * self.lw_prs_wghts)

        # Get higher precisions integral
        high_pres_intg = mid_diff * np.sum(func_eval * self.hg_prs_wghts)

        # Define error
        err = high_pres_intg - low_pres_intg

        # Return value of integral and error
        return high_pres_intg, err, mid_int

    def _adap_gk(self, f, a, b, eps = 1e-5):
        """Adaptive Guass-Kronrod integration

        Description
        -------

        Adaptive Gauss-Kronrod integration. Recursively
        divides integration sub-intervals. Condition for meeting
        tolerance is set by error being less than eps * 100
        percent of high-precision integral's value.


        Attributes:
        -------
           f (func): Function to be integrated
           a (float): Lower bound of integration
           b (float): Upper bound of integration
           whole_integ (float): Integral from a to b
           err (float): Error in integral
           mid_ind (float): Midway point between a and b
        """

        # Initialize by evaluating the whole integral
        whole_integ, err, mid_int = self._gk(f, a, b)

        # Check if error is an instance of gvar and reset if so
        err = err.mean if isinstance(err, gv.GVar) else err

        # Check if integrad is gvar instance
        whl_ntg_bs = whole_integ.mean if isinstance(whole_integ, gv.GVar) else whole_integ

        # Check tolerance
        if abs(err) > abs(eps * whl_ntg_bs):
            # Split interval in half
            return self._adap_gk(f, a, mid_int, eps = eps / 2.) + \
                self._adap_gk(f, mid_int, b, eps = eps / 2.)

        # Return nothing for now
        return whole_integ

    def _adap_gk_nd(self, *args, **kwargs):
        """Adaptive Gauss-Kronrod in n-dimensions

        Description
        -------

        This function wraps the 1d adaptive Gauss-Kronrod integrator
        from Geotecha to perform n-dimensional integrals using Gauss-Kronrod.
        This code is based upon SciPy's implementation of
        n-dimensional integration.


        References
        -------
        [2] https://github.com/scipy/scipy/blob/v1.8.1/scipy/integrate/
            _quadpack_py.py#L694-L825


        Attributes:
        -------
           depth (int): Depth in integral (0 to # dims)
           bound_index (int): Index of boundaries at this depth
           boundaries (array): Boundaries as an array
           a (float): Lower boundary
           b (float): Upper boundary
           f (float): Partialized function representing integrand
                      at this depth.
           intg (float): Result of integration
           err (float): Error in integration
        """

        """ Set integration up """
        # Get depth
        depth = kwargs.pop('depth', 0)

        # Get index of boundary
        bound_index = -(depth + 1)

        # Get function boundaries
        boundaries = self.ranges[bound_index]

        # Get explicit boundaries
        a, b = boundaries[0], boundaries[-1]

        """ Define partialized function, do integration """
        # Check depth
        if (depth + 1 == self.max_depth):
            # Set function to outer layer of user-defined function
            f = np.vectorize(lambda x : self.f(x, *args))
        else: # Set behavior of function to inner-integral
            # Set function to value of integral in inner layer
            f = np.vectorize(functools.partial(self._adap_gk_nd, *args,
                                               depth = depth + 1))

        # Calculate value of integration
        intg = self._adap_gk(f, a, b, eps = self._gk_args['tol'])

        # Return value of integral and error
        return intg

    def _get_gk_abs_wghts(self):
        """Gauss-Kronrod abscissae and weights

        Description
        -------

        Grabs weights and abcissae for Gauss-Kronrod integration. This
        piece of code was adapted from from Geotecha.

        References
        -------
        [1]  https://pythonhosted.org/geotecha/_modules/geotecha/mathematics/
             quadrature.html#gk_quad
        [3]  Holoborodko, Pavel. 'Gauss-Kronrod Quadrature Nodes and
             Weights'.
             http://www.advanpix.com/2011/11/07/gauss-kronrod-quadrature-
             nodes-weights/#Tabulated_Gauss-Kronrod_weights_and_abscissa
        """

        # Define low-order abscissae and weights
        lo_wghts = np.array([[-0.9931285991850949247861224,  0.0176140071391521183118620],
                             [-0.9639719272779137912676661,  0.0406014298003869413310400],
                             [-0.9122344282513259058677524,  0.0626720483341090635695065],
                             [-0.8391169718222188233945291,  0.0832767415767047487247581],
                             [-0.7463319064601507926143051,  0.1019301198172404350367501],
                             [-0.6360536807265150254528367,  0.1181945319615184173123774],
                             [-0.5108670019508270980043641,  0.1316886384491766268984945],
                             [-0.3737060887154195606725482,  0.1420961093183820513292983],
                             [-0.2277858511416450780804962,  0.1491729864726037467878287],
                             [-0.0765265211334973337546404,  0.1527533871307258506980843],
                             [ 0.0765265211334973337546404,  0.1527533871307258506980843],
                             [ 0.2277858511416450780804962,  0.1491729864726037467878287],
                             [ 0.3737060887154195606725482,  0.1420961093183820513292983],
                             [ 0.5108670019508270980043641,  0.1316886384491766268984945],
                             [ 0.6360536807265150254528367,  0.1181945319615184173123774],
                             [ 0.7463319064601507926143051,  0.1019301198172404350367501],
                             [ 0.8391169718222188233945291,  0.0832767415767047487247581],
                             [ 0.9122344282513259058677524,  0.0626720483341090635695065],
                             [ 0.9639719272779137912676661,  0.0406014298003869413310400],
                             [ 0.9931285991850949247861224,  0.0176140071391521183118620]],
                             dtype = float)

        # Define high-order abscissae and weights
        ho_wghts = np.array([[-0.9988590315882776638383156,  0.0030735837185205315012183],
                             [-0.9931285991850949247861224,  0.0086002698556429421986618],
                             [-0.9815078774502502591933430,  0.0146261692569712529837880],
                             [-0.9639719272779137912676661,  0.0203883734612665235980102],
                             [-0.9408226338317547535199827,  0.0258821336049511588345051],
                             [-0.9122344282513259058677524,  0.0312873067770327989585431],
                             [-0.8782768112522819760774430,  0.0366001697582007980305572],
                             [-0.8391169718222188233945291,  0.0416688733279736862637883],
                             [-0.7950414288375511983506388,  0.0464348218674976747202319],
                             [-0.7463319064601507926143051,  0.0509445739237286919327077],
                             [-0.6932376563347513848054907,  0.0551951053482859947448324],
                             [-0.6360536807265150254528367,  0.0591114008806395723749672],
                             [-0.5751404468197103153429460,  0.0626532375547811680258701],
                             [-0.5108670019508270980043641,  0.0658345971336184221115636],
                             [-0.4435931752387251031999922,  0.0686486729285216193456234],
                             [-0.3737060887154195606725482,  0.0710544235534440683057904],
                             [-0.3016278681149130043205554,  0.0730306903327866674951894],
                             [-0.2277858511416450780804962,  0.0745828754004991889865814],
                             [-0.1526054652409226755052202,  0.0757044976845566746595428],
                             [-0.0765265211334973337546404,  0.0763778676720807367055028],
                             [ 0.0000000000000000000000000,  0.0766007119179996564450499],
                             [ 0.0765265211334973337546404,  0.0763778676720807367055028],
                             [ 0.1526054652409226755052202,  0.0757044976845566746595428],
                             [ 0.2277858511416450780804962,  0.0745828754004991889865814],
                             [ 0.3016278681149130043205554,  0.0730306903327866674951894],
                             [ 0.3737060887154195606725482,  0.0710544235534440683057904],
                             [ 0.4435931752387251031999922,  0.0686486729285216193456234],
                             [ 0.5108670019508270980043641,  0.0658345971336184221115636],
                             [ 0.5751404468197103153429460,  0.0626532375547811680258701],
                             [ 0.6360536807265150254528367,  0.0591114008806395723749672],
                             [ 0.6932376563347513848054907,  0.0551951053482859947448324],
                             [ 0.7463319064601507926143051,  0.0509445739237286919327077],
                             [ 0.7950414288375511983506388,  0.0464348218674976747202319],
                             [ 0.8391169718222188233945291,  0.0416688733279736862637883],
                             [ 0.8782768112522819760774430,  0.0366001697582007980305572],
                             [ 0.9122344282513259058677524,  0.0312873067770327989585431],
                             [ 0.9408226338317547535199827,  0.0258821336049511588345051],
                             [ 0.9639719272779137912676661,  0.0203883734612665235980102],
                             [ 0.9815078774502502591933430,  0.0146261692569712529837880],
                             [ 0.9931285991850949247861224,  0.0086002698556429421986618],
                             [ 0.9988590315882776638383156,  0.0030735837185205315012183]],
                             dtype = float)

        # Get abscissae
        self.abscis = ho_wghts[:, 0]

        # Get lo weights
        self.lw_prs_wghts = [0. if ind % 2 == 0 else lo_wghts[(ind - 1) // 2][1]
                             for ind in range(len(ho_wghts))]

        # Get ho weights
        self.hg_prs_wghts = ho_wghts[:, 1]

        # Return nothing
        return None
