import json
import decimal
import numpy as np

class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for numpy data types.
    
    This encoder extends the default JSON encoder to handle numpy data types
    that are not natively serializable by the standard json module. It converts
    numpy integers, floats, arrays, and other types to Python native types.
    
    Used for saving R-Anode model outputs and experimental results in JSON format.
    """

    def default(self, obj):
        """Convert numpy objects to JSON serializable types.
        
        Parameters
        ----------
        obj : object
            The object to serialize
            
        Returns
        -------
        object
            JSON serializable representation of the input object
            
        Raises
        ------
        TypeError
            If the object type is not handled by this encoder
        """
        if isinstance(obj, np.integer):
            return int(obj)

        elif isinstance(obj, np.floating):
            return float(obj)

        elif isinstance(obj, (complex, np.complexfloating)):
            return {"real": obj.real, "imag": obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

ctx = decimal.Context()
ctx.prec = 20

def float_to_str(f):
    """Convert a float to string without scientific notation.
    
    This function ensures that floating point numbers are represented
    as decimal strings rather than scientific notation, which is useful
    for file naming and consistent string representation in R-Anode outputs.
    
    Parameters
    ----------
    f : float
        The floating point number to convert
        
    Returns
    -------
    str
        String representation in decimal format
        
    Notes
    -----
    Uses decimal.Context with 20 decimal places of precision.
    Taken from https://stackoverflow.com/questions/38847690
    """
    d1 = ctx.create_decimal(repr(f))
    return format(d1, 'f')

def str_encode_value(val: float, n_digit=None, formatted=True):
    """Encode a float value into a formatted string for file naming.
    
    This function converts float values into string representations suitable
    for use in file names and parameter encoding. It handles special cases
    like negative zero and provides options for decimal precision control.
    
    Parameters
    ----------
    val : float
        The numerical value to encode
    n_digit : int, optional
        Number of decimal places to use. If None, uses maximum precision
    formatted : bool, default=True
        If True, replaces '.' with 'p' and '-' with 'n' for file-safe names
        
    Returns
    -------
    str
        Encoded string representation of the value
        
    Examples
    --------
    >>> str_encode_value(3.14159, n_digit=2)
    '3p14'
    >>> str_encode_value(-0.5, formatted=True)
    'n0p5'
    """
    if n_digit is not None:
        val_str = '{{:.{}f}}'.format(n_digit).format(val)
    else:
        val_str = float_to_str(val)
    # edge case of negative zero
    if val_str == '-0.0':
        val_str = '0p0'
    
    if formatted:
        val_str = val_str.replace('.', 'p').replace('-', 'n')
    return val_str


def find_zero_crossings(x, y):
    """Find x-values where y crosses zero using linear interpolation.
    
    This function identifies points where a function crosses zero by examining
    sign changes between consecutive data points and using linear interpolation
    to estimate the precise crossing location. Used in R-Anode for finding
    confidence intervals and statistical thresholds.
    
    Parameters
    ----------
    x : array-like
        Independent variable values (must be same length as y)
    y : array-like  
        Dependent variable values (must be same length as x)
        
    Returns
    -------
    list
        List of x-values where y crosses zero
        
    Notes
    -----
    Uses linear interpolation between adjacent points to estimate crossing
    locations more precisely than just using the grid points.
    
    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-2, 2, 100)
    >>> y = x**2 - 1  # crosses zero at x = ±1
    >>> crossings = find_zero_crossings(x, y)
    >>> len(crossings)
    2
    """
    crossings = []
    for i in range(len(y) - 1):
        y0, y1 = y[i], y[i+1]
        if y0 == 0:
            # Exactly zero at i
            crossings.append(x[i])
        elif y1 == 0:
            # Exactly zero at i+1
            crossings.append(x[i+1])
        elif y0 * y1 < 0:
            # There's a sign change between i and i+1
            # Do a linear interpolation for a better estimate
            x0, x1 = x[i], x[i+1]
            crossing_x = x0 - y0 * (x1 - x0) / (y1 - y0)
            crossings.append(crossing_x)
    return crossings
