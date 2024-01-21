from distutils.sysconfig import get_python_inc
from numpy import get_include as npy_get_include

def main():
    python_inc = get_python_inc()
    numpy_inc = npy_get_include()
    print("\nIn order to compile the C functions correctly, add the following\n" + 
        "paths to the environment variable 'C_INCLUDE_PATH':\n")
    print(f"General python C api:\n{python_inc}\n")
    print(f"Numpy C api:\n{numpy_inc}")

if __name__ == "__main__":
    main()

