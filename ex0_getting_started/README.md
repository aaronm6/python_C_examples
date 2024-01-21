# Getting started
---

The C files in these examples will need to be compiled.  Instead of compiling each C file into an executable, in order to be imported as a module in Python, it must be compiled into a shared object file, which has the extension 'so'.  This is true on Linux and MacOS, but not on Windows (see note at the end of this readme).

In order to compile the C code, the Python header file (`Python.h`) needs to be found by the compiler.  Additionally, if numpy arrays are to be passed, manipulated, and returned, the numpy header files must be findable by the compiler.  First, we need to find where these header files live.  The python script `find_c_libraries.py` will find them for the system it is being run on and print to screen.  For example, on my system, I get the following:

<code>
$ python3 find_c_libraries.py

In order to compile the C functions correctly, add the following
paths to the environment variable 'C_INCLUDE_PATH':

General python C api:
/opt/local/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10

Numpy C api:
/opt/local/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/numpy/core/include
</code>



### Note on Windows:
Compiling a C extension for Python works differently on Windows because Windows uses a completely different linking system for shared libraries than Linux/MacOS.  I do not use Windows in my personal and professional life, so I have not bothered trying to figure out how to make Python C extensions in Windows.  I also don't have access to a Windows machine, so trying anything out would be challenging (yes I could create a Windows VM if I **really** wanted to do this, but I just don't have the time).  For the curious reader, [this link](https://docs.python.org/3/extending/windows.html) may be of some help.
