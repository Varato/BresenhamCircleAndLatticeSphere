#include <Python.h>
#include <stdio.h>
#include <stdlib.h>

static PyObject* LatticeSphere3D(PyObject *self, PyObject *args, PyObject *kwds)
{
    long r;
    PyObject *Q1, *arc_length;
    PyObject *ret = PyTuple_New(2); // return a tuple.

    static char *kwlist[] = {(char*)"r", NULL};

    int res = PyArg_ParseTupleAndKeywords(args, kwds, "l", kwlist, &r);
    if(!res) return NULL;

    Q1 = PyList_New(0);
    arc_length = PyList_New(0);
    if (!Q1 || !arc_length) return NULL;

    if (r == 0){
        PyObject *voxel = PyTuple_New(3);
        if (!voxel) return NULL;
        PyTuple_SetItem(voxel, 0, PyLong_FromLong(0));
        PyTuple_SetItem(voxel, 1, PyLong_FromLong(0));
        PyTuple_SetItem(voxel, 2, PyLong_FromLong(0));

        PyList_Append(Q1, voxel);
        PyList_Append(arc_length, PyLong_FromLong(1));

        PyTuple_SetItem(ret, 0, Q1);
        PyTuple_SetItem(ret, 1, arc_length);

        return ret;
    }

    long i = 0, j = 0;
    long k = r, k0 = r;
    long s = 0, s0 = 0;
    long v = r-1, v0 = r-1;
    long l = 2*v0, l0 = 2*v0;
    long al = 0;

    while (i <= k){
        // printf("i = %ld\n", i);
        al = 0;
        while (j <= k){
            if (s > v){
                k = k - 1;
                v = v + l;
                l = l - 2;
            }
            if ( (j <= k) && (s != v || j != k) ){
                PyObject *voxel = PyTuple_New(3);
                if (!voxel) return NULL;
                PyTuple_SetItem(voxel, 0, PyLong_FromLong(i));
                PyTuple_SetItem(voxel, 1, PyLong_FromLong(j));
                PyTuple_SetItem(voxel, 2, PyLong_FromLong(k));
                PyList_Append(Q1, voxel);
                al += 1;
            }
            s = s + 2*j + 1;
            j = j + 1;
        }
        if (al > 0){
            PyList_Append(arc_length, PyLong_FromLong(al));
        }
        s0 = s0 + 4*i + 2;
        i = i + 1;

        while ( s0 > v0 && i <= k0){
            k0 = k0 - 1;
            v0 = v0 + l0;
            l0 = l0 - 2;
        }
        j = i;
        k = k0;
        v = v0;
        l = l0;
        s = s0;
    }

    PyTuple_SetItem(ret, 0, Q1);
    PyTuple_SetItem(ret, 1, arc_length);
    return ret;


}

static PyMethodDef latticeSphere_methods[] = {
    {
        "LatticeSphere3D", (PyCFunction)LatticeSphere3D, METH_VARARGS | METH_KEYWORDS,
        ""
    },
   
    {NULL, NULL, 0, NULL}
};

static PyModuleDef LatticeSphere_kernel = {
    PyModuleDef_HEAD_INIT,
    "LatticeSphere_kernel",
    "",
    -1,
    latticeSphere_methods
};

PyMODINIT_FUNC PyInit_LatticeSphere_kernel(void)
{
    PyObject *LatticeSphere_kernel_mod;
    LatticeSphere_kernel_mod = PyModule_Create(&LatticeSphere_kernel);
    // import_array();
    // import_ufunc();
    return LatticeSphere_kernel_mod;
}
