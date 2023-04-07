#define PY_SSIZE_T_CLEAN
#include <Python.h>

/*
For examples which do a proper, full build using setup.py, see:
https://gist.github.com/physacco/2e1b52415f3a964ad2a542a99bebed8f

*/

static PyObject *method_add_dbs(PyObject *self, PyObject *args) {
	double a, b, c;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }
    c = a + b;
    return PyFloat_FromDouble(c);
}

static PyObject *method_add_li(PyObject *self, PyObject *args) {
	long a, b, c;
	if (!PyArg_ParseTuple(args, "ll",&a, &b)) {
		return NULL;
	}
	c = a + b;
	return PyLong_FromLong(c);
}

static PyObject *method_mult_dbs(PyObject *self, PyObject *args) {
	double a, b, c;
    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "dd", &a, &b)) {
        return NULL;
    }
    c = a * b;
    return PyFloat_FromDouble(c);
}

static PyObject *method_div_li(PyObject *self, PyObject *args) {
	long a, b, c;
	if (!PyArg_ParseTuple(args, "ll", &a, &b)) {
		return NULL;
	}
	c = a/b;
	return PyLong_FromLong(c);
}

static PyMethodDef BinaryMethods[] = {
    {"add_dbl", method_add_dbs, METH_VARARGS, "Add the two things that are doubles"},
    {"add_int", method_add_li, METH_VARARGS, "Add the two things that are ints"},
    {"mult_dbl", method_mult_dbs, METH_VARARGS, "Multiply the two doubles"},
    {"div_li",method_div_li, METH_VARARGS, "Divide two ints"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef binary_module = {
    PyModuleDef_HEAD_INIT,
    "ex2_binaryops",
    "Several binary operations",
    -1,
    BinaryMethods
};

PyMODINIT_FUNC PyInit_ex2_binaryops(void) {
    return PyModule_Create(&binary_module);
}
