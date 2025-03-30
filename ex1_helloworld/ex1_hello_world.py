import os as _os
import sys as _sys
fname_base = 'ex1_hello_world'
so_name = f'{fname_base}.so'

'''
This method assumes that the C code has been compiled separately into a .so file.  If the 
required .so file is not found, it will spit out some instructions on how to compile it 
manually.  The generally preferred way is for distutils methods to handle the compilation
and dependencies, but that's not really the point of these examples.

For examples which do a proper, full build using setup.py, see:
https://gist.github.com/physacco/2e1b52415f3a964ad2a542a99bebed8f
'''

if so_name not in _os.listdir():
    print(f"\n{so_name} does not exist, and must be compiled.  Make sure the following")
    print("paths are added to the C_INCLUDE_PATH environment variable: \n")
    from distutils.sysconfig import get_python_inc
    from numpy import get_include as npy_get_include
    print(get_python_inc())
    print(npy_get_include())
    print("\nCompile the .c file to .so via this command:")
    print(f"* macOS:\n     clang -shared -undefined dynamic_lookup -o {so_name} {fname_base}.c")
    print(f"* Linux:\n     gcc -shared -o {so_name} -fPIC {fname_base}.c")
    print(f"* Windows:\n     Use a different computer. Or google how to compile C code for use in Python")
    print("     Windows doesn't use .so files, so this file will have to change.")
    _sys.exit()

"""
The code below is all you need if the C module is already successfully compiled.  The .so file
must be given where the variable 'so_name' is below.
"""

def __bootstrap__():
    global __bootstrap__, __loader__, __file__
    import sys, pkg_resources, importlib.util
    __file__ = pkg_resources.resource_filename(__name__, so_name)
    __loader__ = None; del __bootstrap__, __loader__
    spec = importlib.util.spec_from_file_location(__name__,__file__)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
__bootstrap__()
