import ex4_ndarrays as ex
import numpy as np

def main():
    """
    This module demonstrates interaction with the numpy C API.
    """
    a_int = np.array([1,2,3], dtype=int)
    a_dbl = np.array([1.,2.,3.], dtype=float)
    a_bol = np.array([0, 1, 1], dtype=bool)
    
    print('---------------- TEST 1 ----------------')
    ex.accept_array(a_int)
    
    print('---------------- TEST 2 ----------------')
    ex.print_dtypes()
    
    print('---------------- TEST 3 ----------------')
    print("Array with dbtype=int:")
    ex.get_dtype(a_int)
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
    temp_arr = ex.add_scalar_to_array(a_dbl, 5.)
    print(f"{a_dbl} + 5. = {temp_arr}")
    
    print('---------------- TEST 8 ----------------')
    temp_arr = ex.multiply_array_by_scalar(a_dbl, 5.)
    print(f"{a_dbl} * 5. = {temp_arr}")
    
    print('---------------- TEST 9 ----------------')
    b_dbl = np.array([11.,12.,13.])
    temp_arr = ex.add_two_arrays(a_dbl, b_dbl)
    print(f"{a_dbl} + {b_dbl} = {temp_arr}")
    
    print('---------------- TEST 10 ----------------')
    c1 = np.array([True, False], dtype=bool)
    c2 = np.array([2.3, 3.4, 4.5])
    c3 = np.array([1,1,1,4,4,4,3,2,3,4,5])
    LL = [c1, c2, c3]
    ex.input_list_of_arrays(LL)
    
    print('---------------- TEST 11 ----------------')
    temp_tuple = ex.create_tuple_of_arrays()
    print(f"{temp_tuple = }")

if __name__ == "__main__":
    main()


