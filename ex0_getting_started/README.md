# Getting started
---

The C files in these examples will need to be compiled.  Instead of compiling each C file into an executable, in order to be imported as a module in Python, it must be compiled into a shared object file, which has the extension `.so`.  This is true on Linux and MacOS, but not on Windows (see note at the end of this readme).

In order to compile the C code, the Python header file (`Python.h`) needs to be found by the compiler.  Additionally, if Numpy arrays are to be passed, manipulated, and returned, the Numpy header files must be findable by the compiler.  First, we need to find where these header files live.  The python script `find_c_libraries.py` will find them for the system it is being run on and print to screen.  For example, on my system, I get the following:

```
$ python3 find_c_libraries.py

In order to compile the C functions correctly, add the following
paths to the environment variable 'C_INCLUDE_PATH':

General python C api:
/opt/local/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10

Numpy C api:
/opt/local/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/numpy/core/include
```

One would then go into the shell's startup script (e.g. `.bashrc`) and add the following lines:
```
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/local/Library/Frameworks/Python.framework/Versions/3.10/include/python3.10
export C_INCLUDE_PATH=$C_INCLUDE_PATH:/opt/local/Library/Frameworks/Python.framework/Versions/3.10/lib/python3.10/site-packages/numpy/core/include
```

In principle, since `gcc` is available on both Linux and MacOS, one command should suffice for compiling on both systems.  Unfortunately, I have not found that to be the case, so I show below how to compile on both systems.  Imagine we have written our C extension module in a file called `my_module.c`.  To compile this to `my_module.so`, one uses:

**macOS:**<br>
```clang -shared -undefined dynamic_lookup -o my_module.so my_module.c```

**Linux:**<br>
```gcc -shared -o my_module.so -fPIC my_module.c```

Typing this *every* single time would be tedious, of course.  I recommend making a bash script called `compileso` to do the job easier:

```
#!/bin/bash

c_filename=$1
filebase="${c_filename%.*}"
so_filename=$filebase.so
clang -shared -undefined dynamic_lookup -o $so_filename $c_filename # for macOS
# gcc -shared -o $so_filename -fPIC c_filename # for linux
echo "...created ${so_filename}"
```
The syntax for compiling `my_module.c` is now:

```$ compileso my_module.c```

which creates `my_module.so` automatically.

### Note on Windows:
Compiling a C extension for Python works differently on Windows because Windows uses a completely different linking system for shared libraries than Linux and MacOS do.  I do not use Windows in my personal and professional life, so I have not bothered trying to figure out how to make Python C extensions in Windows.  I also don't have access to a Windows machine, so trying anything out would be challenging (yes I could create a Windows VM if I *really* wanted to do this, but I just don't have the time).  For the curious reader, [this link](https://docs.python.org/3/extending/windows.html) may be of some help.
