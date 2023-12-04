#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Define the functions in the module
static PyObject *meth_helloworld(PyObject *self, PyObject *Py_UNUSED(b)) {
	printf("Hello World!\n");
	Py_RETURN_NONE;
}

// Define what is in the module
static PyMethodDef HelloMethods[] = {
    {"helloworld", meth_helloworld, METH_NOARGS,"No inputs/outputs, just print hello world\n"},
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef hello_module = {
    PyModuleDef_HEAD_INIT,
    "ex1_hello_world",
    "Provide a function that prints hello world.",
    -1,
    HelloMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_ex1_hello_world(void) {
    return PyModule_Create(&hello_module);
}
