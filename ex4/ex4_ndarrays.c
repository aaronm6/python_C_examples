#define PY_SSIZE_T_CLEAN
#define NUMPY_CORE_INCLUDE_NUMPY_NPY_1_7_DEPRECATED_API_H_
#include <Python.h>
#include <numpy/ndarrayobject.h>
#include "npy_dtypes.h"

/* ----------------- <AUX> ----------------- */
/* npy_dtypes.h contains two array definitions:
	char *type_names[]:
		Array of strings that are the names of the different
		possible dtypes in numpy.  Array is NULL terminated
	int numpy_types[]:
		Array of enumeration integers for the different 
		numpy dtypes.  Array is terminated with -1.
*/
char *get_dtype_name(int dtype_enum) {
	// Multiple type names will match a single dtype integer (run 'print_dtypes'
	// from this module in python).  This function returns only the first hit.
	int i = 0;
	while ((numpy_types[i] >= 0) && (i < 200)) {
		if (dtype_enum == numpy_types[i]) {
			return type_names[i];
		}
		i++;
	}
	return NULL;
}
/* ----------------------- </AUX> ----------------------- */


/* ----------------- <MODULE FUNCTIONS> ----------------- */
static PyObject *meth_accept_array(PyObject *self, PyObject *args) {
	PyArrayObject *input;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &input)) {
		return NULL;
	}
	printf("Successful passing (and releasing) of an ndarray to function!\n");
	fflush(stdout);
	// PyArray_Converter increments the reference count of the input array
	// to indicate that this function is using it.  We must therefore
	// decrement this count at the end of the function to indicate that we
	// are finished, or we will end up with a memory leak.
	Py_DECREF(input);
	Py_RETURN_NONE;
}

static PyObject *meth_accept_array_wrong(PyObject *self, PyObject *args) {
	PyArrayObject *input;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &input)) {
		return NULL;
	}
	printf("Successful passing (but not releasing) of an ndarray to function!\n");
	fflush(stdout);
	// Successful passing of an ndarray to function, but improper DECREF.
	// To see, run sys.getrefcount(..) on the array object passed to this
	// function before and after passing it to this function.\n
	// Then, try the same but using the 'accept_array' function
	// there SHOULD be a PyDECREF(input) line here
	Py_RETURN_NONE;
}

static PyObject *meth_print_dtypes(PyObject *self, PyObject *Py_UNUSED(b)) {
	int i = 0;
	while ((type_names[i] != NULL) && (i < 200)) {
		printf("%s = %i\n", type_names[i], numpy_types[i]);
		i++;
	}
	fflush(stdout);
	Py_RETURN_NONE;
}

static PyObject *meth_get_dtype(PyObject *self, PyObject *args) {
	PyArrayObject *arr;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &arr)){
		return NULL;
	}
	long tp = (long)PyArray_TYPE(arr);
	printf("type integer is %li\n", tp);
	int i = 0;
	while ((numpy_types[i] >= 0) && (i < 200)) {
		if (tp == numpy_types[i]) {printf("%s\n", type_names[i]);}
		i++;
	}
	fflush(stdout);
	Py_DECREF(arr);
	Py_RETURN_NONE;
}

static PyObject *meth_describe_array(PyObject *self, PyObject *args) {
	PyArrayObject *arr;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &arr)) {
		return NULL;
	}
	int ndim = PyArray_NDIM(arr);
	npy_intp *pydims = PyArray_DIMS(arr);
	npy_intp *strds = PyArray_STRIDES(arr);
	long tp = (long)PyArray_TYPE(arr);
	int flags = PyArray_FLAGS(arr);
	printf("    dtype: %s\n", get_dtype_name(tp));
	if (PyArray_IS_C_CONTIGUOUS(arr)) {
		printf("    C-contiguous\n");
	}
	if (PyArray_IS_F_CONTIGUOUS(arr)) {
		printf("    F-contiguous\n");
	}
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
	printf("    Data pointer:   %p\n", PyArray_DATA(arr));
	printf("    Object pointer: %p\n", (void *)arr);
	fflush(stdout);
	Py_DECREF(arr);
	Py_RETURN_NONE;
}

static PyObject *meth_create_float_array(PyObject *self, PyObject *Py_UNUSED(b)) {
	int ndim = 1; // 1-dimensional array
	npy_intp dims[ndim]; // declare the array of dimensions
	dims[0] = 3; // array of length 3
	PyObject *nd_outarr = PyArray_EMPTY(ndim, dims, NPY_DOUBLE, NPY_CORDER); // define and declare array object
	npy_double *outarr = (npy_double *)PyArray_DATA(nd_outarr); // pointer to first element in data
	// Note the naming scheme used here: 'nd_outarr' is the python array object (which, by the way,
	// does not include the data of the array).  'outarr' is the pointer to the actual data in the
	// array.
	outarr[0] = 3.14159;
	outarr[1] = 2.71828;
	outarr[2] = 1.41421;
	// The method of indexing the data array used here is not the 
	// 'proper' way, and can be problematic.  Here it is fine, but
	// later a safer method will be used.
	return nd_outarr;
}

static PyObject *meth_create_float_array2(PyObject *self, PyObject *Py_UNUSED(b)) {
	int ndim = 1; // 1-dimensional array
	npy_intp dims[ndim]; // declare the array of dimensions
	dims[0] = 3; // array of length 3
	PyArray_Descr *sc_type = PyArray_DescrNewFromType(NPY_FLOAT64);
	PyObject *nd_outarr = PyArray_Empty(ndim, dims, sc_type, NPY_CORDER);
	// Note the different method here of declarying the empty array ('EMPTY' vs 'Empty',
	// which require different methods).  Both are acceptable.
	npy_float64 *el;
	// PyArray__GETPTR1 returns a pointer to a specified element in the 1d array.  It
	// is safer than dereferencing the data pointer directly, as in the above function.
	el = (npy_float64 *)PyArray_GETPTR1(nd_outarr, 0);
	*el = 3.14159;
	el = (npy_float64 *)PyArray_GETPTR1(nd_outarr, 1);
	*el = 2.71828;
	el = (npy_float64 *)PyArray_GETPTR1(nd_outarr, 2);
	*el = 1.41421;
	
	return nd_outarr;
}


static PyObject *meth_create_bool_array(PyObject *self, PyObject *Py_UNUSED(b)) {
	// C has no native bool type, but python does.
	int ndim = 1;
	npy_intp dims[1];
	dims[0] = 3;
	PyObject *nd_outarr = PyArray_EMPTY(ndim, dims, NPY_BOOL, NPY_CORDER);
	npy_intp *strds_bytes = PyArray_STRIDES(nd_outarr);
	npy_bool *outarr = (npy_bool *)PyArray_DATA(nd_outarr);
	outarr[0] = NPY_FALSE;
	outarr[1] = NPY_TRUE;
	outarr[2] = NPY_TRUE;
	// The method of indexing the data array used here is not the 
	// 'proper' way, and can be problematic.  Here it is fine, but
	// later the technique of using the 'strides' will be explained.
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
	PyObject *nd_outarr = PyArray_ZEROS(ndim, dims, NPY_LONG, NPY_CORDER);
	npy_long *outarr = (npy_long *)PyArray_DATA(nd_outarr);
	for (long i=0L;i<degree*degree; i++) {
		outarr[i] = i + 1L;
	}
	// The method of indexing the data array used here is not the 
	// 'proper' way, and can be problematic.  Here it is fine, but
	// later the technique of using the 'strides' will be explained.
	return nd_outarr;
}

static PyObject *meth_copy_1d_int8_array_wrong(PyObject *self, PyObject *args) {
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&",PyArray_Converter,&nd_x)){
		return NULL;
	}
	if (PyArray_TYPE(nd_x) != NPY_INT8) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array must be char/int8");
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 1-dimensional");
		return NULL;
	}
	PyObject *nd_out = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	npy_int8 *x = (npy_int8 *)PyArray_DATA(nd_x);
	npy_int8 *out = (npy_int8 *)PyArray_DATA(nd_out);
	for (npy_intp i=0; i<numEl; i++) {
		out[i] = x[i];
	}
	Py_DECREF(nd_x);
	return nd_out;
}

static PyObject *meth_copy_1d_int8_array_right(PyObject *self, PyObject *args) {
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&",PyArray_Converter,&nd_x)){
		return NULL;
	}
	if (PyArray_TYPE(nd_x) != NPY_INT8) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array must be char/int8");
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 1-dimensional");
		return NULL;
	}
	//itemsize is the number of bytes of a single element in the array.
	//In this case, we have int8, so that is 1 byte per element.  If we
	//were instead dealing with e.g. float64, that is 8 bytes, so itemsize 
	//would be 8.
	npy_intp el_size_bytes = PyArray_ITEMSIZE(nd_x);

	PyObject *nd_out = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	npy_int8 *x = (npy_int8 *)PyArray_DATA(nd_x);
	npy_intp *x_strds = PyArray_STRIDES(nd_x);
	// the 'strides' are the number of bytes between each element in the
	// data of the array.  If the array is contiguous in memory, 1-dim, 
	// and the dtype is an 8-bit object (like int8), then the strides 
	// will be a 1-element array, with 1.  If the same, but the dtype is 
	// e.g. a float64 (i.e. 64 bits = 8 bytes), then there will be 8 bytes
	// between each element, if contiguous.  If not contiguous, then 
	// strides will indicate that.
	
	
	npy_int8 *out = (npy_int8 *)PyArray_DATA(nd_out);
	npy_intp *out_strds = PyArray_STRIDES(nd_out);
	
	for (npy_intp i=0; i<numEl; i++) {
		out[i*out_strds[0]/el_size_bytes] = x[i*x_strds[0]/el_size_bytes];
	}
	//The above assumes that the itemsize of nd_out and nd_x are the same
	Py_DECREF(nd_x);
	return nd_out;
}

static PyObject *meth_copy_1d_int8_array_get(PyObject *self, PyObject *args) {
	// In this case, we access the data by using Numpy's native 'PyArray_GETPTR1',
	// rather than directly scanning through the data array
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&",PyArray_Converter,&nd_x)){
		return NULL;
	}
	if (PyArray_TYPE(nd_x) != NPY_INT8) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array must be char/int8");
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 1) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 1-dimensional");
		return NULL;
	}
	//npy_intp el_size_bytes = PyArray_ITEMSIZE(nd_x);

	PyObject *nd_out = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	npy_int8 *x_el, *out_el;
	for (npy_intp i=0; i<numEl; i++) {
		x_el = (npy_int8 *)PyArray_GETPTR1(nd_x,i);
		out_el = (npy_int8 *)PyArray_GETPTR1(nd_out,i);
		out_el[0] = x_el[0];
	}
	Py_DECREF(nd_x);
	return nd_out;
}
static PyObject *meth_copy_2d_float_array_wrong(PyObject *self, PyObject *args) {
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &nd_x)) {
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 2-dimensional");
	}
	if (PyArray_TYPE(nd_x) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array must have a dtype of double/float64");
	}
	//npy_intp *dims = PyArray_DIMS;
	PyObject *nd_y = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	
	npy_float64 *x = (npy_float64 *)PyArray_DATA(nd_x);
	npy_float64 *y = (npy_float64 *)PyArray_DATA(nd_y);
	for (int i=0; i<numEl; i++) {
		y[i] = x[i];
	}
	Py_DECREF(nd_x);
	return nd_y;
}
static PyObject *meth_copy_2d_float_array_right(PyObject *self, PyObject *args) {
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &nd_x)) {
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 2-dimensional");
	}
	npy_intp *dims = PyArray_DIMS(nd_x);
	if (PyArray_TYPE(nd_x) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array must have a dtype of double/float64");
	}
	//npy_intp *dims = PyArray_DIMS;
	PyObject *nd_y = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	
	npy_float64 *x = (npy_float64 *)PyArray_DATA(nd_x);
	npy_float64 *y = (npy_float64 *)PyArray_DATA(nd_y);
	npy_intp *x_strides_bytes = PyArray_STRIDES(nd_x);
	npy_intp *y_strides_bytes = PyArray_STRIDES(nd_y);
	npy_intp itemsize = PyArray_ITEMSIZE(nd_x); // size of an element of x in bytes
	npy_intp x_str[ndim];
	npy_intp y_str[ndim];
	for (int k=0; k<ndim; k++) {
		x_str[k] = x_strides_bytes[k] / itemsize;
		y_str[k] = y_strides_bytes[k] / itemsize;
	}
	npy_intp i, j, k_x, k_y;
	for (npy_intp el=0; el<numEl; el++) {
		i = el % dims[1];
		j = el / dims[1];
		k_x = i*x_str[1] + j*x_str[0];
		k_y = i*y_str[1] + j*y_str[0];
		y[k_y] = x[k_x];
	}
	Py_DECREF(nd_x);
	return nd_y;
}
static PyObject *meth_copy_2d_float_array_get(PyObject *self, PyObject *args) { // loop through els, calc inds
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &nd_x)) {
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 2-dimensional");
	}
	if (PyArray_TYPE(nd_x) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array must have a dtype of double/float64");
	}
	npy_intp *dims = PyArray_DIMS(nd_x);
	PyObject *nd_y = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	npy_intp numEl_y = PyArray_SIZE(nd_y);
	npy_float64 *x_el, *y_el;
	
	npy_intp i, j; // column index, row index
	for (int el=0; el<numEl; el++) {
		i = el % dims[1];
		j = el / dims[1];
		x_el = (npy_float64 *)PyArray_GETPTR2(nd_x, j, i);
		y_el = (npy_float64 *)PyArray_GETPTR2(nd_y, j, i);
		*y_el = *x_el;
	}
	Py_DECREF(nd_x);
	return nd_y;
}
static PyObject *meth_copy_2d_float_array_get2(PyObject *self, PyObject *args) { // direct loop over inds
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&", PyArray_Converter, &nd_x)) {
		return NULL;
	}
	int ndim = PyArray_NDIM(nd_x);
	if (ndim != 2) {
		PyErr_SetString(PyExc_TypeError, "Input array must be 2-dimensional");
	}
	if (PyArray_TYPE(nd_x) != NPY_FLOAT64) {
		PyErr_SetString(PyExc_TypeError, "Input array must have a dtype of double/float64");
	}
	npy_intp *dims = PyArray_DIMS(nd_x);
	PyObject *nd_y = PyArray_NewLikeArray(nd_x, NPY_ANYORDER, NULL, 1);
	npy_intp numEl = PyArray_SIZE(nd_x);
	npy_float64 *x_el, *y_el;
	
	for (npy_intp j=0; j<dims[0]; j++) {
		for (npy_intp i=0; i<dims[1]; i++) {
			x_el = (npy_float64 *)PyArray_GETPTR2(nd_x, j, i);
			y_el = (npy_float64 *)PyArray_GETPTR2(nd_y, j, i);
			y_el[0] = x_el[0];
		}
	}
	Py_DECREF(nd_x);
	return nd_y;
}
/*
shape: (x, y, z), el_strides(a, b, c);

[[[ 1, 2, 3, 4],
  [ 5, 6, 7, 8],
  [ 9,10,11,12]],
 
 [[13,14,15,16],
  [17,18,19,20],
  [21,22,23,24]]]

 for n in range(24):
    ...:     #M = a*int(n/(y*z))*(y*z) + b*int(n/z)%y + c*n%z
    ...:     ax1 = int(n/(y*z))
    ...:     ax2 = int(n/z)%y
    ...:     ax3 = n%z
    ...:     M = ax1*a + ax2*b + ax3*c
    ...:     print(f'{M = }:  {ax1 = }; {ax2 = }; {ax3 = }')
*/
/*
static PyObject *meth_copy_nd_double_array(PyObject *self, PyObject *args) {
	PyArrayObject *nd_x;
	if (!PyArg_ParseTuple(args, "O&",PyArray_Converter,&nd_x)){
		return NULL;
	}
	if (PyArray_TYPE(nd_x) != NPY_DOUBLE) {
		PyErr_SetString(PyExc_TypeError, "dtype of input array must be double/float64");
		return NULL;
	}
	npy_double *x = (npy_double *)PyArray_DATA(nd_x);
	int ndim = PyArray_NDIM(nd_x);
	npy_intp numel = PyArray_SIZE(nd_x);
	npy_intp *dims = PyArray_DIMS(nd_x);
	npy_intp *xstr_bytes = PyArray_STRIDES(nd_x);
	npy_intp el_size_bytes = PyArray_ITEMSIZE(nd_x);
	npy_intp xstr[ndim];
	for (int i=0; i<ndim; i++) {
		xstr[i] = xstr_bytes[i] / el_size_bytes;
	}
	npy_intp inds[ndim];
	for (int i=0; i<ndim; i++) {
		inds[i] = 0;
	}
	
	PyObject *nd_y = PyArray_NewLikeArray(nd_x, NPY_CORDER, NULL, 1);
	npy_double *y = (npy_double *)PyArray_DATA(nd_y);
	npy_intp ystr_bytes = PyArray_STRIDES(nd_y);
	npy_intp ystr[ndim];
	for (int i=0; i<ndim; i++) {
		ystr[i] = ystr_bytes[i] / el_size_bytes;
	}
	
	npy_intp x_addr, y_addr;
	int inc_dim = 0;
	//while (el < numel) {
	for (int el=0; el<numel; el++) {
		// do stuff
		x_addr = 0;
		y_addr = 0;
		for (int j=0; j<ndim; j++) {
			//go through the inds and calculate the address of the element
		}
		inds[ndims-inc_dim-1] = (inds[ndims-inc_dim-1] + 1) % dims[ndims-inc_dim-1];
		while (inds[ndims-inc_dim-1] == 0) {
			inc_dim++;
			inds[ndims-inc_dim-1] = (inds[ndims-inc_dim-1] + 1) % dims[ndims-inc_dim-1];
		}
	}
}
*/
// From the numpy ref manual:
// The correct way to access the itemsize of an array from the C API is:
//    PyArray_ITEMSIZE(arr)
/*
static PyObject *meth_add_scalar_to_array_wrong(PyObject *self, PyObject *args) {
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
*/
/*
	int ndim = PyArray_NDIM(input);
	npy_intp *pydims = PyArray_DIMS(input);
	npy_intp *strds = PyArray_STRIDES(input);
*/
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
	accept_array_wrong__doc__,
	"accept_array_wrong(x)\n--\n\n"
	"Same as the function 'accept_array', but this one does not handle\n"
	"the reference counting correctly.");
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
	"No input.  Create a numpy array of dtype double (i.e. numpy type float64\n"
	"(on most platforms) and return it. The internal C code does not do it in\n"
	"the 'proper' way, but shown here as a demonstration.");
PyDoc_STRVAR(
	create_float_array2__doc__,
	"create_float_array2()\n--\n\n"
	"No input.  Create a numpy array of dtype double (i.e. numpy type float64\n"
	"(on most platforms) and return it. Here the internal C code accesses the\n"
	"array elements in a way that is safer than that used in 'create_float_array'");
PyDoc_STRVAR(
	create_bool_array__doc__,
	"create_bool_array()\n--\n\n"
	"No input.  Create numpy array of python type bool.  This is a separate\n"
	"function from 'create_float_array' because C has no boolean data type,\n"
	"so one has to use the typdef from the numpy API.");
PyDoc_STRVAR(
	create_square_array__doc__,
	"create_square_array(n)\n--\n\n"
	"Create a square array of python int (numpy int64) type.  The input, n,\n"
	"which defines the size; i.e. a n-x-n square array is produced.");
PyDoc_STRVAR(
	copy_1d_int8_array_wrong__doc__,
	"copy_1d_int8_array_wrong(a)\n--\n\n"
	"Provide an array of type int8 and return a copy of that\n"
	"array.  This is the 'wrong' way to do it because it will barf if\n"
	"the input array is a slice of another array.\n"
	"To create an int8 array, one can use 'astype', for example:\n"
	">>> a = np.array([1,2,3,4,5]).astype(np.int8)");
PyDoc_STRVAR(
	copy_1d_int8_array_right__doc__,
	"copy_1d_int8_array_right(a)\n--\n\n"
	"Provide an array of type int8 and return a copy of that\n"
	"array.  This is the 'right' way to do it because it will\n"
	"properly handle the array 'strides', which is important\n"
	"if the input array is a slice of another array.\n"
	"To create an int8 array, one can use 'astype', for example:\n"
	">>> a = np.array([1,2,3,4,5]).astype(np.int8)");
PyDoc_STRVAR(
	copy_1d_int8_array_get__doc__,
	"copy_1d_int8_array_get(a)\n--\n\n"
	"Provide an array of type int8 and return a copy of that\n"
	"array.  This is also a 'right' way to do it, and uses\n"
	"Numpy's 'PyArray_GETPTR1' function for accessing the data.\n"
	"The functionality on the python side should be exactly the\n"
	"same as for 'copy_1d_int8_array_right'.");
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
	{"accept_array_wrong", meth_accept_array_wrong, METH_VARARGS, accept_array_wrong__doc__},
	{"print_dtypes", meth_print_dtypes, METH_NOARGS, print_dtypes__doc__},
	{"get_dtype", meth_get_dtype, METH_VARARGS, get_dtype__doc__},
	{"describe_array",meth_describe_array,METH_VARARGS, describe_array__doc__},
	{"create_float_array",meth_create_float_array,METH_NOARGS,create_float_array__doc__},
	{"create_float_array2",meth_create_float_array2,METH_NOARGS,create_float_array2__doc__},
	{"create_bool_array",meth_create_bool_array,METH_NOARGS,create_bool_array__doc__},
	{"create_square_array",meth_create_square_array,METH_VARARGS,create_square_array__doc__},
	{"copy_1d_int8_array_wrong",meth_copy_1d_int8_array_wrong,METH_VARARGS,copy_1d_int8_array_wrong__doc__},
	{"copy_1d_int8_array_right",meth_copy_1d_int8_array_right,METH_VARARGS,copy_1d_int8_array_right__doc__},
	{"copy_1d_int8_array_get",meth_copy_1d_int8_array_get,METH_VARARGS,copy_1d_int8_array_get__doc__},
	{"copy_2d_float_array_wrong",meth_copy_2d_float_array_wrong,METH_VARARGS,"copy 2d wrong"},
	{"copy_2d_float_array_right",meth_copy_2d_float_array_right,METH_VARARGS,"copy 2d right"},
	{"copy_2d_float_array_get",meth_copy_2d_float_array_get,METH_VARARGS,"copy 2d get"},
	{"copy_2d_float_array_get2",meth_copy_2d_float_array_get2,METH_VARARGS,"copy 2d get2"},
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





