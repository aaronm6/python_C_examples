import ex3_passlist as ex
from time import perf_counter
import numpy as np
from numpy.random import rand

def main():
    """
    ex3_passlist has two functions to test (and two that can be ignored):
        lsum: accepts a python list of numbers (ints or floats, but must all be the same)
              and returns their sum as a single number.
        ldouble: accepts a python list of numbers (ints or floats, but must all be the same)
              and returns a list of the same length that is the element-wise double of the 
              original list.
    Both of these functions have variants with a "_nc" suffix.  These are just tests for how
    to extract the values in the list and do math on them.  The "_nc" variants appear to 
    be slower.
    """
    a = [3,5,7,23,87]
    a_sum = ex.lsum(a)
    a_dbl = ex.ldouble(a)
    print(f"Sum of {a} is {a_sum}")
    print(f"Double of {a} is {a_dbl}")
    
    nLongList = int(1e6)
    b = (100*rand(nLongList)).astype(int).tolist()
    b_arr = np.array(b)
    t_start = perf_counter()
    b_sum = ex.lsum(b)
    t_stop = perf_counter()
    tt_sum = t_stop - t_start
    t_start = perf_counter()
    b_sum_nc = ex.lsum_nc(b)
    t_stop = perf_counter()
    tt_sum_nc = t_stop - t_start
    t_start = perf_counter()
    b_sum_native = sum(b)
    t_stop = perf_counter()
    tt_sum_native = t_stop - t_start
    t_start = perf_counter()
    b_sum_np = b_arr.sum()
    t_stop = perf_counter()
    tt_sum_np = t_stop - t_start
    t_start = perf_counter()
    b_dbl = ex.ldouble(b)
    t_stop = perf_counter()
    tt_dbl = t_stop - t_start
    t_start = perf_counter()
    b_dbl_nc = ex.ldouble_nc(b)
    t_stop = perf_counter()
    tt_dbl_nc = t_stop - t_start
    t_start = perf_counter()
    b_dbl_native = [2*item for item in b]
    t_stop = perf_counter()
    tt_dbl_native = t_stop - t_start
    t_start = perf_counter()
    b_dbl_np = 2 * b_arr
    t_stop = perf_counter()
    tt_dbl_np = t_stop - t_start
    
    print(f"Sum of {nLongList:.0e} elements with convert: {tt_sum:e} s")
    print(f"Sum of {nLongList:.0e} elements w/o  convert: {tt_sum_nc:e} s")
    print(f"Sum of {nLongList:.0e} elements natively    : {tt_sum_native:e} s")
    print(f"Sum of {nLongList:.0e} elements w/ numpy    : {tt_sum_np:e} s")
    
    print(f"Doubling {nLongList:.0e} els with convert: {tt_dbl:e} s")
    print(f"Doubling {nLongList:.0e} els w/o  convert: {tt_dbl_nc:e} s")
    print(f"Doubling {nLongList:.0e} els w/ list comp: {tt_dbl_native:e} s")
    print(f"Doubling {nLongList:.0e} els w/ list comp: {tt_dbl_np:e} s")
    
    created_list = ex.create_list()
    print(f'{created_list = }')
    created_tuple = ex.create_tuple()
    print(f'{created_tuple = }')


if __name__ == "__main__":
    main()






