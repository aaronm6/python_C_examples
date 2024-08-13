#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
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
static PyObject *meth_createlist(PyObject *self, PyObject *Py_UNUSED(args)) {
	// Create the list: [1, 2, 'three']
	PyObject *my_list;
	my_list = PyList_New(3);
	PyList_SetItem(my_list, 0, PyLong_FromLong(1L));
	PyList_SetItem(my_list, 1, PyLong_FromLong(2L));
	PyList_SetItem(my_list, 2, PyUnicode_FromString("three"));
	return my_list;
}

static PyObject *meth_createtuple(PyObject *self, PyObject *Py_UNUSED(args)) {
	// Create the tuple: (1, 2, 'three')
	PyObject *my_tuple;
	my_tuple = PyTuple_New(3);
	PyObject *a = PyLong_FromLong(1L);
	PyObject *c = PyUnicode_FromString("three");
	PyTuple_SetItem(my_tuple, 0, a);
	PyTuple_SetItem(my_tuple, 1, PyLong_FromLong(2L));
	PyTuple_SetItem(my_tuple, 2, c);
	return my_tuple;
}

static PyObject *method_describe_args(PyObject *self, PyObject *args) {
	PyTypeObject *args_type = Py_TYPE(args);
	PyObject *type_name = PyObject_GetAttrString((PyObject*)args_type, "__name__");
	printf("'args's type name is: '%s'\n", PyUnicode_AsUTF8(type_name));
	if (Py_IS_TYPE(args, &PyTuple_Type)) {
		Py_ssize_t args_size = PyTuple_Size(args);
		printf("%zd positional arguments were given.\n", args_size);
	}
	fflush(stdout);
	Py_RETURN_NONE;
}


static PyObject *meth_list_sum(PyObject *self, PyObject *args) {
	/*Take a list of python ints and calculate the sum*/
	int numElements; /* how many lines we passed for parsing */
	
	PyObject *listobj; /* the list of numbers */
	PyObject *elobj; /* pointer to the element in the string */
	/* Parse arguments.  The '!' after the 'O' means it should cast
	   the input PyObject to be of the type given by the preceeding
	   type (i.e. PyList_Type, which has to be given by reference).
	*/
	if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &listobj)) {
		return NULL;
	}
	numElements = PyList_Size(listobj);
	if (numElements < 0) {
		PyErr_SetString(PyExc_ValueError, "List must not have negative length.");
	}
	long c = 0L;  /* c will hold the sum */
	
	for (int i=0; i<numElements; i++) {
		elobj = PyList_GetItem(listobj, i);
		// Notice here we are converting the list item to a C object
		// and then doing the math with C objects (i.e. 'c += ...'
		c += PyLong_AsLong(elobj);
	}
	return PyLong_FromLong(c);
}

static PyObject *meth_list_sum_nc(PyObject *self, PyObject *args) {
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
	long c = 0L;
	PyObject * py_c = PyLong_FromLong(c);
	for (int i=0; i<numElements; i++) {
		elobj = PyList_GetItem(listobj, i);
		// Notice here we are not converting the list item to a C object.
		// We do the math on the C objects using an API function.
		py_c = PyNumber_Add(py_c, elobj);
	}
	return py_c;
}

static PyObject *meth_list_double(PyObject *self, PyObject *args) {
	/* Take a list of python ints and create a copy where each element
	is double of its counterpart element */
	Py_ssize_t list_size;
	PyObject *in_list, *out_list; // the list objects
	PyObject *elobj_i, *elobj_o; // the element objects
	
	long n_i, n_o;
	if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &in_list)) {
		return NULL;
	}
	// The above line ('PyList_Type') implicitly checks that the given
	// object is of type list.
	list_size = PyList_Size(in_list);
	out_list = PyList_New(list_size);
	for (int i=0; i<list_size; i++) {
		elobj_i = PyList_GetItem(in_list, i);
		// Below, we convert the eleobj to a C object do the math
		// on C objects, convert back to a PyObject and then insert
		// that into the new list.
		n_i = PyLong_AsLong(elobj_i);
		n_o = 2 * n_i;
		elobj_o = PyLong_FromLong(n_o);
		PyList_SetItem(out_list, i, elobj_o);
	}
	return out_list;
}

static PyObject *meth_list_double_nc(PyObject *self, PyObject *args){
	/* 'nc' = 'no convert'.Do the same thing as list_double, but don't convert the 
	list items to C long ints before doing the math.  Trying to do the math on the 
	PyObjects themselves... maybe this is faster? */
	Py_ssize_t list_size;
	PyObject *in_list, *out_list;
	PyObject *elobj_i, *elobj_o;
	PyObject * pylong2 = PyLong_FromLong(2L);
	
	long n_i, n_o;
	if(!PyArg_ParseTuple(args, "O!", &PyList_Type, &in_list)) {
		return NULL;
	}
	
	list_size = PyList_Size(in_list);
	out_list = PyList_New(list_size);
	for (int i=0; i<list_size; i++) {
		elobj_i = PyList_GetItem(in_list, i);
		// Below we do not convert the list elements to C objects,
		// and instead do the math on the PyObjects themselves
		// using functions from the Python API.
		elobj_o = PyNumber_Multiply(pylong2, elobj_i);
		PyList_SetItem(out_list, i, elobj_o);
	}
	return out_list;
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
	describe_args__doc__,
	"describe_args(*args)\n--\n\n"
	"Accept a variable number of positional arguments and describe that tuple.");
PyDoc_STRVAR(
	list_sum__doc__,
	"list_sum(x)\n--\n\n"
	"Given a list, x, of exclusively python ints (C longs), calculate the sum\n"
	"of the elements and return as a python int.\n"
	"Internally, the code converts the elements from python objects to\n"
	"type C long, performs the sum on the C objects, and converts the sum\n"
	"to a python int object.");
PyDoc_STRVAR(
	list_sum_nc__doc__,
	"list_sum_nc(x)\n--\n\n"
	"Same as lsum(x), but internally the math is done with the python objects directly.\n"
	"'_nc' stands for 'no conversion' i.e. conversion to C objects.");
PyDoc_STRVAR(
	list_x2__doc__,
	"list_x2(x)\n--\n\n"
	"Given a list, x, of elements that are python ints this returns another list\n"
	"in which each element of the returned list is double its counterpart element\n"
	"in x.  Elements of the input list must all be python ints (C longs)");
PyDoc_STRVAR(
	list_x2_nc__doc__,
	"list_x2_nc(x)\n--\n\n"
	"Same as ldouble(x), but internally the math is done with python objects directly.\n"
	"'_nc' stands for 'no conversion' i.e. conversion to C objects.");
/* ----------------- </DOC STRINGS> ----------------- */

static PyMethodDef ListMethods[] = {
    {"create_list", meth_createlist, METH_NOARGS, create_list__doc__},
    {"create_tuple", meth_createtuple, METH_NOARGS, create_tuple__doc__},
    {"describe_args",method_describe_args, METH_VARARGS, describe_args__doc__},
    {"list_sum", meth_list_sum, METH_VARARGS, list_sum__doc__},
    {"list_sum_nc", meth_list_sum_nc, METH_VARARGS, list_sum_nc__doc__},
    {"list_x2", meth_list_double, METH_VARARGS, list_x2__doc__},
    {"list_x2_nc", meth_list_double_nc, METH_VARARGS,list_x2_nc__doc__},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef list_module = {
    PyModuleDef_HEAD_INIT,
    "ex3_lists",
    "Pass and create python lists",
    -1,
    ListMethods
};

PyMODINIT_FUNC PyInit_ex3_lists(void) {
    return PyModule_Create(&list_module);
}
