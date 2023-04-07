import ex2_binaryops as ex

def main():
    """
    ex2_binaryops has functions:
        add_dbl(a,b) -> accepts two floats (doubles), returns their sum
        add_int(a,b) -> accepts two ints (longs), returns their sum
        mult_dbl(a,b)-> accepts two floats, returns their product
        div_li(a,b)-> accept two ints, returns their integer division
    """
    a_dbs, b_dbs = 3., 5.
    a_int, b_int = 3, 5
    print(f"{a_dbs} + {b_dbs} = {ex.add_dbl(a_dbs,b_dbs)}")
    print(f"{a_int} + {b_int} = {ex.add_int(a_int,b_int)}")
    print(f"{a_dbs} * {b_dbs} = {ex.mult_dbl(a_dbs,b_dbs)}")
    print(f"{a_int}/{b_int} = {ex.div_li(a_int,b_int)}")
    

if __name__ == "__main__":
    main()
