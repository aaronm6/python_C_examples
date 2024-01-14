#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Define the functions in the module
static PyObject *meth_helloworld(PyObject *self, PyObject *Py_UNUSED(b)) {
	// Print the message with the native C printf
	printf("Hello World!\n");

	// A function that returns nothing has to actually return None, 
	// with the following command:
	Py_RETURN_NONE;
}

static PyObject *meth_helloworld_jupyter(PyObject *self, PyObject *Py_UNUSED(b)) {
	// printf somehow prints without using stdout; it prints to the terminal only.
	// Therefore it can't be used in a Jupyter notebook.  This function uses Python's
	// builtin 'print' function to accomplish this; it is a little more complicated
	// than 'meth_helloworld' as a consequence.
	
	// Get access to the builtin functions
	PyObject *builtins = PyEval_GetBuiltins();
	
	// Find the 'print' function from the builtins
	PyObject *pyprint = PyDict_GetItemString(builtins, "print");
	
	// Call the print function and pass 'Hello World' to it
	PyObject *a = PyObject_CallOneArg(pyprint, PyUnicode_FromString("Hello World!"));
	
	// Decrement the reference count of 'a' since we are done with it now
	Py_DECREF(a);
	Py_RETURN_NONE;
}

// Define what is in the module
static PyMethodDef HelloMethods[] = {
	{"helloworld", meth_helloworld, METH_NOARGS,"No inputs/outputs, just print hello world\n"},
	{"helloworld_jupyter", meth_helloworld_jupyter, METH_NOARGS, "hello world that works in a Jupyter notebook\n"},
	{NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef hello_module = {
	PyModuleDef_HEAD_INIT,
	"ex1_hello_world",
	"Provide two functions that print hello world.",
	-1,
	HelloMethods
};

// Initialize the module
PyMODINIT_FUNC PyInit_ex1_hello_world(void) {
	return PyModule_Create(&hello_module);
}
