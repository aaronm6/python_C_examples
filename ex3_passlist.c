#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
This module just includes some tests involving functions that accept python lists
as input arguments.  There are two functions:

1. Return the sum of the (integer) elements of the input list
2. Return a copy of a list where each element is the double of each element of the input list

Both functions have two copies: one where the elements are converted to C types before doing
math on them, the other where math is done directly on the PyObject* items, using things like
PyNumber_Add and PyNumber_Multiply.

So far, it seems the versions that do math 'directly' on the PyObjects is slower.

This module includes some examples of passing and returning list objects (and
a tuple).  There are two copies of the following functions:
	sum: add up all the elements in a list and return the sum
	double: return a copy of a list in which each element is double the element in the input list
There are two copies of each of these two functions because there are two ways doing math on the 
elements of the list:
	1: Convert the list elements into C types and do the math with normal C operations
	2: Use Python.h functions to perform math operations on the list elements.  E.g.
		PyNumber_Add(.,.)  returns a python object that is the sum of the two input python objects
		PyNumber_Multiply(.,.) -> same as above but for multiplication
The versions of the functions which uses (2) above are labeled with a '_nc' suffix, which stands for
'no convert'.  The two methods are chosen so as to enable speed tests.  It seems that converting
the objects to C types and using C math operations is faster than using PyNumber_ functions.

Additionally, there are two functions which take no inputs but create a (1) list, and (2) tuple.
*/

/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *createlist(PyObject *self, PyObject *Py_UNUSED(b)) {
	PyObject *t;
	t = PyList_New(3);
	PyList_SetItem(t, 0, PyLong_FromLong(1L));
	PyList_SetItem(t, 1, PyLong_FromLong(2L));
	PyList_SetItem(t, 2, PyUnicode_FromString("three"));
	
	return t;
}

static PyObject *createtuple(PyObject *self, PyObject *Py_UNUSED(b)) {
	PyObject *t;
	t = PyTuple_New(3);
	PyObject *c = PyUnicode_FromString("three");
	PyTuple_SetItem(t, 0, PyLong_FromLong(1L));
	PyTuple_SetItem(t, 1, PyLong_FromLong(2L));
	PyTuple_SetItem(t, 2, c);
	//Py_DECREF(c); <---- DO NOT DECREF
	return t;
}

static PyObject *list_sum(PyObject *self, PyObject *args) {
	/*Take a list of python ints and calculate the sum*/
	int numElements; /* how many lines we passed for parsing */
	
	PyObject *listobj; /* the list of numbers */
	PyObject *elobj; /* pointer to the element in the string */
	long int n;
	//Py_ssize_t x = listobj->ob_refcnt;
	//printf("******** List's ref count right after declaration is: %li\n", x);
	
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &listobj)) {
        return NULL;
    }
	numElements = PyList_Size(listobj);
	if (numElements < 0) {
		return NULL;
	}
	long int c = 0;  /* c will hold the sum */
	for (int i=0; i<numElements; i++) {
		elobj = PyList_GetItem(listobj, i);
		n = PyLong_AsLong(elobj);
		c += n;
	}
	Py_DECREF(elobj);
    return PyLong_FromLong(c);
}

static PyObject *list_sum_nc(PyObject *self, PyObject *args) {
	/* Same as list_sum, but don't convert every item to a C object before doing 
	math.  I.e. do the math on the python objects directly (which might just
	be doing the conversion itself under the hood, who knows) */
	int numElements; /* how many lines we passed for parsing */
	
	PyObject * listobj; /* the list of numbers */
	PyObject * elobj; /* pointer to the element in the string */
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &listobj)) {
        return NULL;
    }
	numElements = PyList_Size(listobj);
	long int c=0;
	PyObject * py_c = PyLong_FromLong(c);
	for (int i=0; i<numElements; i++) {
		elobj = PyList_GetItem(listobj, i);
		py_c = PyNumber_Add(py_c, elobj);
	}
	return py_c;
}

static PyObject *list_double(PyObject *self, PyObject *args) {
	/* Take a list of python ints and create a copy where each element
	is double of its counterpart element */
	Py_ssize_t list_size;
	PyObject *listobj;
	PyObject *newList;
	PyObject * elobj;
	PyObject * elobj2;
	
	long int n, n2;
	if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &listobj)) {
		return NULL;
	}
	
	list_size = PyList_Size(listobj);
	newList = PyList_New(list_size);
	for (int i=0; i<list_size; i++) {
		elobj = PyList_GetItem(listobj, i);
		n = PyLong_AsLong(elobj);
		n2 = 2 * n;
		elobj2 = PyLong_FromLong(n2);
		PyList_SetItem(newList, i, elobj2);
	}
	return newList;
}

static PyObject *list_double_nc(PyObject *self, PyObject *args){
	/* 'nc' = 'no convert'.Do the same thing as list_double, but don't convert the 
	list items to C long ints before doing the math.  Trying to do the math on the 
	PyObjects themselves... maybe this is faster? */
	Py_ssize_t list_size;
	PyObject *listobj;
	PyObject *newList;
	PyObject * elobj;
	PyObject * elobj2;
	long int long2 = 2;
	PyObject * pylong2 = PyLong_FromLong(long2);
	
	long int n, n2;
	if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &listobj)) {
		return NULL;
	}
	
	list_size = PyList_Size(listobj);
	newList = PyList_New(list_size);
	for (int i=0; i<list_size; i++) {
		elobj = PyList_GetItem(listobj, i);
		elobj2 = PyNumber_Multiply(pylong2, elobj);
		PyList_SetItem(newList, i, elobj2);
	}
	return newList;
}
/* ----------------- </MODULE FUNCTIONS> ----------------- */

/* ----------------- <DOC STRINGS> ----------------- */
PyDoc_STRVAR(
	create_list__doc__,
	"create_list()\n--\n\n"
	"Create the following list and return it: [1, 2, 'three']");

PyDoc_STRVAR(
	create_tuple__doc__,
	"create_tuple()\n--\n\n"
	"Create the following tuple and return it: (1, 2, 'three')");

PyDoc_STRVAR(
	lsum__doc__,
	"lsum(x)\n--\n\n"
	"Given a list, x, of exclusively python ints (C longs), calculate the sum\n"
	"of the elements and return as a python int.\n"
	"Internally, the code converts the elements from python objects to\n"
	"type C long, performs the sum on the C objects, and converts the sum\n"
	"to a python int object.");
PyDoc_STRVAR(
	lsum_nc__doc__,
	"lsum_nc(x)\n--\n\n"
	"Same as lsum(x), but internally the math is done with the python objects directly.\n"
	"'_nc' stands for 'no conversion' i.e. conversion to C objects.");
PyDoc_STRVAR(
	ldouble__doc__,
	"ldouble(x)\n--\n\n"
	"Given a list, x, this returns another list in which each element of the returned\n"
	"list is double its counterpart element in x.  Elements of the input list must\n"
	"all be python ints (C longs)");
PyDoc_STRVAR(
	ldouble_nc__doc__,
	"ldouble_nc(x)\n--\n\n"
	"Same as ldouble(x), but internally the math is done with python objects directly.\n"
	"'_nc' stands for 'no conversion' i.e. conversion to C objects.");
/* ----------------- </DOC STRINGS> ----------------- */

static PyMethodDef ListMethods[] = {
    {"create_list", createlist, METH_NOARGS, create_list__doc__},
    {"create_tuple", createtuple, METH_NOARGS, create_tuple__doc__},
    {"lsum", list_sum, METH_VARARGS, lsum__doc__},
    {"lsum_nc", list_sum_nc, METH_VARARGS, lsum_nc__doc__},
    {"ldouble", list_double, METH_VARARGS, ldouble__doc__},
    {"ldouble_nc", list_double_nc, METH_VARARGS,ldouble_nc__doc__},
    {NULL, NULL, 0, NULL}
};

/*     {"create_list", createlistmeth, METH_NOARGS, "Create [1, 2, 'three']"},
*/

static struct PyModuleDef list_module = {
    PyModuleDef_HEAD_INIT,
    "ex3_passlist",
    "Pass a list to a function",
    -1,
    ListMethods
};

PyMODINIT_FUNC PyInit_ex3_passlist(void) {
    return PyModule_Create(&list_module);
}
