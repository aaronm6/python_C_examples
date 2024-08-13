#define PY_SSIZE_T_CLEAN
#include <Python.h>

// Define the functions in the module
static PyObject *meth_helloworld(PyObject *self, PyObject *Py_UNUSED(args)) {
	// Print the message with the native C printf
	printf("Hello World!\n");
	
	// For some reason, printf will not work on its own when executed in
	// a Jupyter notebook (though it will do fine if run from the terminal).
	// The 'fflush' function forces it to display from a Jupyter notebook.
	fflush(stdout);
	
	// A function that returns nothing has to actually return None, 
	// with the following command:
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

// Initialize the module. The part after 'PyInit_' must be the same
// as the name of the module ('ex1_hello_world') that you will type.
// That is 'import ex1_hello_world'
PyMODINIT_FUNC PyInit_ex1_hello_world(void) {
	return PyModule_Create(&hello_module);
}
