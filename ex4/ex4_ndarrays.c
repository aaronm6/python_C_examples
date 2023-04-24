#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>

/* ----------------- </MODULE FUNCTIONS> ----------------- */
static PyObject *meth_accept_array(PyObject *self, PyObject *args) {
	PyArrayObject *input;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &input)) {
		return NULL;
	}
	printf("Successful passing of an ndarray to function!\n");
	Py_DECREF(input);
	Py_RETURN_NONE;
}

static PyObject *meth_print_dtypes(PyObject *self, PyObject *Py_UNUSED(b)) {
	printf("Numpy dtype integer codes:\n");
	printf("            --\n");
	printf("NPY_BOOL is %i\n", NPY_BOOL);
	printf("NPY_INT8 is %i\n", NPY_INT8);
	printf("NPY_INT32 is %i\n", NPY_INT32);
	printf("NPY_INT64 is %i\n", NPY_INT64);
	printf("NPY_INT is %i\n", NPY_INT);
	printf("NPY_LONG is %i\n", NPY_LONG);
	printf("NPY_FLOAT is %i\n", NPY_FLOAT);
	printf("NPY_DOUBLE is %i\n", NPY_DOUBLE);
	printf("NPY_FLOAT64 is %i\n", NPY_FLOAT64);
	printf("(value) NPY_TRUE is %i\n", NPY_TRUE);
	printf("(value) NPY_FALSE is %i\n", NPY_FALSE);
	Py_RETURN_NONE;
}

static PyObject *meth_get_dtype(PyObject *self, PyObject *args) {
	PyArrayObject *arr;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &arr)){
		return NULL;
	}
	long tp = (long)PyArray_TYPE(arr);
	if (tp == NPY_BOOL) {printf("NPY_BOOL\n");}
	if (tp == NPY_INT8) {printf("NPY_INT8\n");}
	if (tp == NPY_INT32) {printf("NPY_INT32\n");}
	if (tp == NPY_INT64) {printf("NPY_INT64\n");}
	if (tp == NPY_INT) {printf("NPY_INT\n");}
	if (tp == NPY_LONG) {printf("NPY_LONG\n");}
	if (tp == NPY_FLOAT) {printf("NPY_FLOAT\n");}
	if (tp == NPY_DOUBLE) {printf("NPY_DOUBLE\n");}
	if (tp == NPY_FLOAT64) {printf("NPY_FLOAT64\n");}
	Py_DECREF(arr);
	Py_RETURN_NONE;
}

static PyObject *meth_describe_array(PyObject *self, PyObject *args) {
	PyArrayObject *input;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &input)) {
		return NULL;
	}
	int ndim = PyArray_NDIM(input);
	npy_intp *pydims = PyArray_DIMS(input);
	npy_intp *strds = PyArray_STRIDES(input);
	printf("    Number of dimensions: %i\n", ndim);
	printf("    Shape = (");
	for (npy_intp i=0; i<ndim; i++) {
		printf("%li", pydims[i]);
		if ((i+1)<ndim) { printf(", ");}
	}
	printf(")\n");
	printf("    Strides: (");
	for (npy_intp i=0; i<ndim; i++) {
		printf("%li", strds[i]);
		if ((i+1)<ndim) { printf(", ");}
	}
	printf(")\n");
	printf("    Data pointer:   %p\n", PyArray_DATA(input));
	printf("    Object pointer: %p\n", (void *)input);
	Py_DECREF(input);
	Py_RETURN_NONE;
}

static PyObject *meth_create_float_array(PyObject *self, PyObject *Py_UNUSED(b)) {
	int ndim = 1;
	npy_intp dims[1];
	dims[0] = 3;
	int ft = NPY_CORDER;
	int tp = NPY_DOUBLE;
	PyObject *outarr = PyArray_EMPTY(ndim, dims, tp, ft);
	npy_intp *strds_bytes = PyArray_STRIDES(outarr);
	double *outarr_ptr = (double *)PyArray_DATA(outarr);
	npy_intp strds_el = strds_bytes[0] / sizeof(outarr_ptr[0]);
	outarr_ptr[0*strds_el] = 3.14159;
	outarr_ptr[1*strds_el] = 2.71828;
	outarr_ptr[2*strds_el] = 1.41421;
	return outarr;
}

static PyObject *meth_create_bool_array(PyObject *self, PyObject *Py_UNUSED(b)) {
	// C has no native bool type, but python does.
	int ndim = 1;
	npy_intp dims[1];
	dims[0] = 3;
	PyObject *nd_outarr = PyArray_EMPTY(ndim, dims, NPY_BOOL, NPY_CORDER);
	npy_intp *strds_bytes = PyArray_STRIDES(nd_outarr);
	npy_bool *outarr = (npy_bool *)PyArray_DATA(nd_outarr);
	npy_intp strds_el = strds_bytes[0] / sizeof(outarr[0]);
	outarr[0] = NPY_FALSE;
	outarr[1] = NPY_TRUE;
	outarr[2] = NPY_TRUE;
	return nd_outarr;
}

static PyObject *meth_create_square_array(PyObject *self, PyObject *args) {
	long degree;
	if(!PyArg_ParseTuple(args, "l", &degree)) {
		return NULL;
	}
	int ndim = 2;
	npy_intp dims[2];
	dims[0] = degree;
	dims[1] = degree;
	PyObject *outarr = PyArray_EMPTY(ndim, dims, NPY_LONG, NPY_CORDER);
	long *outarr_ptr = (long *)PyArray_DATA(outarr);
	for (long i=0L;i<(degree*degree);i++) {
		outarr_ptr[i] = i;
	}
	return outarr;
}

static PyObject *meth_add_scalar_to_array(PyObject *self, PyObject *args) {
	double n;
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&d",PyArray_Converter,&nd_x,&n)){
		return NULL;
	}
	int x_tp = PyArray_TYPE(nd_x);
	if (x_tp != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array 'x' must be double/float64");
		return NULL;
	}
	PyObject *nd_out = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	double *x = (double *)PyArray_DATA(nd_x);
	double *out = (double *)PyArray_DATA(nd_out);
	for (npy_intp i=0; i<numEl; i++) {
		out[i] = x[i] + n;
	}
	Py_DECREF(nd_x);
	return nd_out;
}

static PyObject *meth_multiply_array_by_scalar(PyObject *self, PyObject *args) {
	double n;
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&d",PyArray_Converter,&nd_x,&n)){
		return NULL;
	}
	int x_tp = PyArray_TYPE(nd_x);
	if (x_tp != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array 'x' must be double/float64");
		return NULL;
	}
	PyObject *nd_out = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	double *x = (double *)PyArray_DATA(nd_x);
	double *out = (double *)PyArray_DATA(nd_out);
	for (npy_intp i=0; i<numEl; i++) {
		out[i] = x[i] * n;
	}
	Py_DECREF(nd_x);
	return nd_out;
}

static PyObject *meth_add_two_arrays(PyObject *self, PyObject *args) {
	PyArrayObject *nd_x, *nd_y;
	if (!PyArg_ParseTuple(args, "O&O&",PyArray_Converter,&nd_x,PyArray_Converter,&nd_y)){
		return NULL;
	}
	int x_tp = PyArray_TYPE(nd_x);
	int y_tp = PyArray_TYPE(nd_y);
	if (x_tp != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array 'x' must be double/float64");
		return NULL;
	}
	if (y_tp != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array 'y' must be double/float64");
		return NULL;
	}
	npy_intp x_size = PyArray_SIZE(nd_x);
	npy_intp y_size = PyArray_SIZE(nd_y);
	if (x_size != y_size) {
		PyErr_SetString(PyExc_IndexError, "Arrays 'x' and 'y' must have the same number of elements");
		return NULL;
	}
	PyObject *nd_out = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	double *x = (double *)PyArray_DATA(nd_x);
	double *y = (double *)PyArray_DATA(nd_y);
	double *out = (double *)PyArray_DATA(nd_out);
	for (npy_intp i=0; i<numEl; i++) {
		out[i] = x[i] + y[i];
	}
	Py_DECREF(nd_x);
	Py_DECREF(nd_y);
	return nd_out;
}

static PyObject *meth_input_list_of_arrays(PyObject *self, PyObject *args) {
	PyObject *listobj;
	//PyArrayObject *el_ptr;
	PyObject *el_ptr;
	//double *el;
	
	if(!PyArg_ParseTuple
	(args, "O!", &PyList_Type, &listobj)) {
		return NULL;
	}
	int numEl = PyList_Size(listobj);
	npy_intp arraySize;
	for (int i=0; i<numEl; i++) {
		el_ptr = PyList_GetItem(listobj, i);
		arraySize = PyArray_SIZE(el_ptr);
		printf("Item %i has %li elements\n", i, arraySize);
	}
	//DO NOT DO THIS: Py_DECREF(listobj);
	Py_RETURN_NONE;
}

static PyObject *meth_create_tuple_of_arrays(PyObject *self, PyObject *Py_UNUSED(b)) {
	int ndim = 1;
	npy_intp dims[1];
	dims[0] = 3;
	PyObject *nd_outarr1 = PyArray_EMPTY(ndim, dims, NPY_DOUBLE, NPY_CORDER);
	double *outarr1 = (double *)PyArray_DATA(nd_outarr1);
	outarr1[0] = 3.14159;
	outarr1[1] = 2.71828;
	outarr1[2] = 1.41421;
	PyObject *nd_outarr2 = PyArray_EMPTY(ndim, dims, NPY_LONG, NPY_CORDER);
	long *outarr2 = (long *)PyArray_DATA(nd_outarr2);
	outarr2[0] = 27L;
	outarr2[1] = 54L;
	outarr2[2] = 108L;
	
	// create a tuple and fill it with the two ndarrays that were created above
	PyObject *out_tuple = PyTuple_New(2);
	PyTuple_SetItem(out_tuple, 0, nd_outarr1);
	PyTuple_SetItem(out_tuple, 1, nd_outarr2);
	//Py_DECREF(nd_outarr1); --->  DO NOT DECREF THIS (UNLIKE IN A DICT)
	
	return out_tuple;
}

/* ----------------- <MODULE FUNCTIONS> ----------------- */

/* ----------------- <DOC STRINGS> ----------------- */
PyDoc_STRVAR(
	accept_array__doc__,
	"accept_array(x)\n--\n\n"
	"Accept a numpy array as an input and do nothing (besides print\n"
	"on success).  This simply tests if the right #include worked and\n"
	"the 'import_array()' line is in the PyMODINIT_FUNC definition.");
PyDoc_STRVAR(
	print_dtypes__doc__,
	"print_dtypes()\n--\n\n"
	"Numpy has a code system for describing the data type of elements\n"
	"in an array.  Each data type has an integer that represents it.\n"
	"The numpy API has a list of variables that handle this, so the user\n"
	"never has to write the integers directly (like 'NPY_DOUBLE'). This\n"
	"function takes no input, gives no output, and gives a list of data\n"
	"types, their");
PyDoc_STRVAR(
	get_dtype__doc__,
	"get_dtype(x)\n--\n\n"
	"Input a numpy array and the function prints the corresponding data type.\n"
	"No output.");
PyDoc_STRVAR(
	describe_array__doc__,
	"describe_array(arr)\n--\n\n"
	"Input a numpy array and describe its properties like size and strides.");
PyDoc_STRVAR(
	create_float_array__doc__,
	"create_float_array()\n--\n\n"
	"No input.  Create a numpy array of dtype C double / python float\n"
	"and return it.");
PyDoc_STRVAR(
	create_bool_array__doc__,
	"create_bool_array()\n--\n\n"
	"No input.  Create numpy array of python type bool.  This is a separate\n"
	"function from 'create_float_array' because C has no boolean data type,\n"
	"so one has to use the typdef from the numpy API.");
PyDoc_STRVAR(
	create_square_array__doc__,
	"create_square_array(n)\n--\n\n"
	"Create a square array of python int type.  The input, n, which\n"
	"defines the size; i.e. a n-x-n square array is produced.");
PyDoc_STRVAR(
	add_scalar_to_array__doc__,
	"add_scalar_to_array(x,n)\n--\n\n"
	"x is a numpy array of dtype numpy.float64 / python float\n"
	"n is a python scalar of dtype float.\n"
	"Return x+n");
PyDoc_STRVAR(
	multiply_array_by_scalar__doc__,
	"multiply_array_by_scalar(x,n)\n--\n\n"
	"x is a numpy array of dtype numpy.float64 / python float\n"
	"n is a python scalar of dtype float.\n"
	"Return x*n");
PyDoc_STRVAR(
	add_two_arrays__doc__,
	"add_two_arrays(x,y)\n--\n\n"
	"Input two numpy arrays, (x and y) of dtype float.\n"
	"Return a new array that is the element-wise sum of the two.");
PyDoc_STRVAR(
	input_list_of_arrays__doc__,
	"input_list_of_arrays(L)\n--\n\n"
	"Give a python list (L) whose elements are numpy arrays.  This function\n"
	"unpacks the list and prints the size of each array. Returns nothing.");
PyDoc_STRVAR(
	create_tuple_of_arrays__doc__,
	"create_tuple_of_arrays()\n--\n\n"
	"No inputs.  Creates two numpy arrays and fills them into a python tuple.\n"
	"Returns the tuple.");
/* ----------------- </DOC STRINGS> ----------------- */


static PyMethodDef ArrayMethods[] = {
	{"accept_array", meth_accept_array, METH_VARARGS, accept_array__doc__},
	{"print_dtypes", meth_print_dtypes, METH_NOARGS, print_dtypes__doc__},
	{"get_dtype", meth_get_dtype, METH_VARARGS, get_dtype__doc__},
	{"describe_array",meth_describe_array,METH_VARARGS, describe_array__doc__},
	{"create_float_array",meth_create_float_array,METH_NOARGS,create_float_array__doc__},
	{"create_bool_array",meth_create_bool_array,METH_NOARGS,create_bool_array__doc__},
	{"create_square_array",meth_create_square_array,METH_VARARGS,create_square_array__doc__},
	{"add_scalar_to_array",meth_add_scalar_to_array,METH_VARARGS,add_scalar_to_array__doc__},
	{"multiply_array_by_scalar",meth_multiply_array_by_scalar,METH_VARARGS,multiply_array_by_scalar__doc__},
	{"add_two_arrays",meth_add_two_arrays,METH_VARARGS,add_two_arrays__doc__},
	{"input_list_of_arrays",meth_input_list_of_arrays,METH_VARARGS,input_list_of_arrays__doc__},
	{"create_tuple_of_arrays",meth_create_tuple_of_arrays,METH_NOARGS, create_tuple_of_arrays__doc__},
	{NULL, NULL, 0, NULL}
};

/* the following struct defines properties of the module itself */
static struct PyModuleDef array_module = {
	PyModuleDef_HEAD_INIT,
	"ex4_ndarrays",
	"Pass a numpy ndarray to some functions, and/or create a numpy array with some functions.",
	-1,
	ArrayMethods
};

/* NOTE: in the function below, 'import_array()' must be included, which does not exist in the other
   other examples that use python-only API functions and variable types.

The name of the function of type PyMODINIT_FUNC has to be "PyInit_{name}", where {name} is the name
of the module as it will be imported from python, and has to match the secend element of the module
struct defined above.
 */
PyMODINIT_FUNC PyInit_ex4_ndarrays(void) {
	import_array();
	return PyModule_Create(&array_module);
}





