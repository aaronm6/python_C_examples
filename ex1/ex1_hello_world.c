#define PY_SSIZE_T_CLEAN
#include <Python.h>

static PyObject *meth_helloworld(PyObject *self, PyObject *Py_UNUSED(b)) {
	printf("Hello World!\n");
	Py_RETURN_NONE;
}

static PyMethodDef HelloMethods[] = {
    {"helloworld", meth_helloworld, METH_NOARGS,"No inputs/outputs, just print hello world\n"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef hello_module = {
    PyModuleDef_HEAD_INIT,
    "ex1_hello_world",
    "Provide a function that prints hello world.",
    -1,
    HelloMethods
};

PyMODINIT_FUNC PyInit_ex1_hello_world(void) {
    return PyModule_Create(&hello_module);
}
