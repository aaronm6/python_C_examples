#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *method_return_long(PyObject *self, PyObject *Py_UNUSED(b)) {
	long n = 262144L;
	PyObject *n_py = PyLong_FromLong(n);
	return n_py;
}

static PyObject *method_accept_1_int_v1(PyObject *self, PyObject *args) {
	PyObject *a;
	/* PyArg_ParseTuple unpacks the input "args" a bit.  "O" tells it what to 
	   do with the input args, in this case, treat it as a python object (NOT 
	   a C object).  Later, the object is converted to a C long type, which 
	   can be manipulated in C (after checking its type). */
	if (!PyArg_ParseTuple(args, "O", &a)) {
		return NULL;
	}
	// Verify that the type of a is long, but do nothing if not.
	if (!Py_IS_TYPE(a, &PyLong_Type)) {
		printf("Input given must be of [python] type int (a long in C)\n");
	}
	/*
	Convert the python object 'a' to a C object 'n'. This function will
	barf if a's type is not a python int.
	*/
	long n = PyLong_AsLong(a);
	printf("Input given is: %li\n", n);
	Py_RETURN_NONE;
}

static PyObject *method_accept_1_int_v2(PyObject *self, PyObject *args) {
	long n;
	/* PyArg_ParseTuple unpacks the input "args" a bit.  "l" tells it what to 
	   do with the input args, in this case, convert it directly into a C
	   object (declared ahead of time).  Because we specify "l", this function
	   will barf if the given variable is not a long int (python int).*/
	if (!PyArg_ParseTuple(args, "l", &n)) {
		return NULL;
	}
	printf("Input given is: %li\n", n);
	Py_RETURN_NONE;
}

static PyObject *method_check_type(PyObject *self, PyObject *args) {
	PyObject *a;
	if (!PyArg_ParseTuple(args, "O", &a)) {
		return NULL;
	}
	if (Py_IS_TYPE(a, &PyLong_Type)) {
		long a_long = PyLong_AsLong(a);
		printf("Input is %li, of type PyLong\n", a_long);
	} else if (Py_IS_TYPE(a, &PyFloat_Type)) {
		double a_double = PyFloat_AsDouble(a);
		printf("Input is %f, of type PyFloat\n", a_double);
	} else if (Py_IS_TYPE(a, &PyUnicode_Type)) {
		const char *a_string = PyUnicode_AsUTF8(a);
		printf("Input is '%s', of type PyUnicode (i.e. string)\n", a_string);
	} else {
		printf("Input is of some other type\n");
	}
	PyObject *a_type = (PyObject *)Py_TYPE(a);
	PyObject *type_name = PyObject_GetAttrString(a_type, "__name__");
	printf("Object's type name is: '%s'\n", PyUnicode_AsUTF8(type_name));
	
	Py_RETURN_NONE;
}

static PyObject *method_compare_string(PyObject *self, PyObject *args) {
	PyObject *a;
	if (!PyArg_ParseTuple(args, "O", &a)) {
		return NULL;
	}
	int compare_status;
	const char compare_string[] = "default";
	if (!Py_IS_TYPE(a, &PyUnicode_Type)) {
		PyErr_SetString(PyExc_TypeError, "Input must be a string.");
	}
	compare_status = PyUnicode_CompareWithASCIIString(a, compare_string);
	if (compare_status == 0) {
		printf("Input '%s' IS the same as '%s'\n", PyUnicode_AsUTF8(a), compare_string);
	} else {
		printf("Input '%s' IS NOT the same as '%s'\n", PyUnicode_AsUTF8(a), compare_string);
	}
	Py_RETURN_NONE;
}

static PyObject *method_add_floats(PyObject *self, PyObject *args) {
	// Add two numbers that are C doubles (or floats in python)
	double a, b; // the inputs
	double c; // the outputs

	if(!PyArg_ParseTuple(args, "dd", &a, &b)) {
		return NULL;
	}
	c = a + b;
	return PyFloat_FromDouble(c);
}

static PyMethodDef FunctionMethods[] = {
	{"return_long",method_return_long, METH_NOARGS, "Returns a C long, which is a python int. No input."},
	{"accept_1_int_v1",method_accept_1_int_v1, METH_VARARGS, "Accepts a python int and prints. No output."},
	{"accept_1_int_v2",method_accept_1_int_v2, METH_VARARGS, "Accepts a python int and prints. No output."},
	{"check_type",method_check_type, METH_VARARGS, "Check one input is either float, int, or string"},
	{"compare_string", method_compare_string, METH_VARARGS, "Check if given string is 'default'."},
	{"add_floats", method_add_floats, METH_VARARGS, "Add the two things that are [python] floats"},
	{NULL, NULL, 0, NULL}
};

static struct PyModuleDef FunctionModule = {
	PyModuleDef_HEAD_INIT,
	"ex2_basic_funcs",
	"Methods that do some basic functions.",
	-1,
	FunctionMethods
};

PyMODINIT_FUNC PyInit_ex2_basic_funcs(void) {
	return PyModule_Create(&FunctionModule);
}
