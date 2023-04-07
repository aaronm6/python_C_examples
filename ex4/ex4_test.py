import ex4_ndarrays as ex
import numpy as np
import sys

def main():
    """
    This module demonstrates interaction with the numpy C API.
    
    One must manage memory when writing C extensions.  Python's garbage collector works by
    keeping a tally on how many references there are to each particular variable.  If a python
    variable's reference count drops to zero, the garbage collector performs a 'free' operation
    on its memory.  The interpreter increments the reference count of a variable whenever a 
    variable is passed to a function.  It is the responsibility of that function to then 
    decrement the reference count of that variable by the time that function finishes.  This
    is handled automatically for function written in python, but not for python functions 
    written in C.  In ex4_ndarrays.c, notice that a Py_DECREF call is usually given at the 
    end of each module function.  These need to be given for any PyObject given to the function
    as an input.  We don't do it for ints or float given as input, because PyArg_ParseTuple
    unpacks those for use as C variables.  But this must be respected for PyObjects used 
    inside C, like PyArrayObjects (i.e. numpy arrays).  We don't PyDECREF PyObjects that are
    returned by the C function, because that variable will then be used outside of the C function
    and therefore we don't want its refcount to drop to zero.
    
    In the python code below, we use sys.getrefcount to make sure that the reference count of 
    variables before and after a C function stays the same.  This is important.  If a C function
    is called repeatedly from python, and the reference count isn't handled correctly, one can
    easily develop a severe memory leak (if things are Py_DECREF'd correctly) or alternatively
    a segfault (if things are Py_DECREF'd too aggressively).
    
    Lists and dicts containing python variables are particularly confusing.  A variable added to 
    a dict object gets its refcount incremented by the python interpreter, while a variable 
    added to a list does not.  This is an added layer of trickiness; in the next example dealing
    with dicts, we'll show that the variables in a dict need to be Py_DECREF'd after they are
    added.
    """
    a_int = np.array([1,2,3], dtype=int)
    a_dbl = np.array([1.,2.,3.], dtype=float)
    a_bol = np.array([0, 1, 1], dtype=bool)
    
    print('---------------- TEST 1 ----------------')
    print(f"a_int refcount is {sys.getrefcount(a_int)} before function call")
    ex.accept_array(a_int)
    print(f"a_int refcount is {sys.getrefcount(a_int)} after function call")
    
    print('---------------- TEST 2 ----------------')
    ex.print_dtypes()
    
    print('---------------- TEST 3 ----------------')
    print("Array with dbtype=int:")
    print(f"a_int refcount is {sys.getrefcount(a_int)} before function call")
    ex.get_dtype(a_int)
    print(f"a_int refcount is {sys.getrefcount(a_int)} after function call")
    print("Array with dbtype=float:")
    ex.get_dtype(a_dbl)
    print("Array with dbtype=bool:")
    ex.get_dtype(a_bol)
    
    print('---------------- TEST 4 ----------------')
    temp_arr = ex.create_float_array()
    print(f"{temp_arr = }")
    
    print('---------------- TEST 5 ----------------')
    temp_arr = ex.create_bool_array()
    print(f"{temp_arr = }")
    
    print('---------------- TEST 6 ----------------')
    temp_arr = ex.create_square_array(4)
    print(f"temp_arr = \n{temp_arr}")
    
    print('---------------- TEST 7 ----------------')
    print(f"a_dbl refcount is {sys.getrefcount(a_dbl)} before function call")
    temp_arr = ex.add_scalar_to_array(a_dbl, 5.)
    print(f"a_dbl refcount is {sys.getrefcount(a_dbl)} after function call")
    print(f"{a_dbl} + 5. = {temp_arr}")
    
    print('---------------- TEST 8 ----------------')
    print(f"a_dbl refcount is {sys.getrefcount(a_dbl)} before function call")
    temp_arr = ex.multiply_array_by_scalar(a_dbl, 5.)
    print(f"a_dbl refcount is {sys.getrefcount(a_dbl)} after function call")
    print(f"{a_dbl} * 5. = {temp_arr}")
    
    print('---------------- TEST 9 ----------------')
    b_dbl = np.array([11.,12.,13.])
    print(f"a_dbl, b_dbl refcounts are {sys.getrefcount(a_dbl)}, {sys.getrefcount(b_dbl)}, " + 
        "respectively (before function call)")
    temp_arr = ex.add_two_arrays(a_dbl, b_dbl)
    print(f"a_dbl, b_dbl refcounts are {sys.getrefcount(a_dbl)}, {sys.getrefcount(b_dbl)}, " + 
        "respectively (after function call)")
    print(f"{a_dbl} + {b_dbl} = {temp_arr}")
    
    print('---------------- TEST 10 ----------------')
    c1 = np.array([True, False], dtype=bool)
    c2 = np.array([2.3, 3.4, 4.5])
    c3 = np.array([1,1,1,4,4,4,3,2,3,4,5])
    print(f"c1, c2, c3 refcounts are {sys.getrefcount(c1)}, {sys.getrefcount(c2)}, {sys.getrefcount(c3)} "+
        "before function call and list insertion")
    LL = [c1, c2, c3]
    print(f"LL's refcount is {sys.getrefcount(LL)}  (LL = [c1, c2, c3]) before function call")
    print(f"c1, c2, c3 refcounts are {sys.getrefcount(c1)}, {sys.getrefcount(c2)}, {sys.getrefcount(c3)} "+
        "before function call and after list insertion")
    ex.input_list_of_arrays(LL)
    print(f"LL's refcount is {sys.getrefcount(LL)} after function call")
    print(f"c1, c2, c3 refcounts are {sys.getrefcount(c1)}, {sys.getrefcount(c2)}, {sys.getrefcount(c3)} "+
        "after function call")
    
    print('---------------- TEST 11 ----------------')
    temp_tuple = ex.create_tuple_of_arrays()
    print(f"{temp_tuple = }")

if __name__ == "__main__":
    main()


