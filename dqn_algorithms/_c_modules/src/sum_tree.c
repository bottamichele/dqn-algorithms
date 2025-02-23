#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <structmember.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>

#include <math.h>


/* ========================================
 * =============== SUM TREE ===============
 * ======================================== */

typedef struct {
    PyObject_HEAD
    int n_leaves;           //Number of concrete leaves of tree.
    int depth;              //Depth of tree.
    int total_nodes;        //Total number of nodes of tree.
    PyObject* tree;         //Complete binary tree.
} SumTree;


/* ========================================
 * =========== USEFUL FUNCTIONS ===========
 * ========================================*/

/* Check if a node index is correct.
 * @param self a SumTree instance.
 * @param idx_node a node index.
 * @return 1 if node index is correct, 0 otherwise. */
static int checkNodeIndex(SumTree* self, int idx_node) {
    return idx_node >= 0 && idx_node < self->total_nodes;
}

/* Get parent index of a node.
 * @param self a SumTree instance.
 * @param idx_node a node index.
 * @param idx_parent parent node index of idx_node.
 * @return 1 if this function does not raise any exception, 0 otherwise. */
static int getParentIndex(SumTree* self, int idx_node, int* idx_parent) {
    if(!checkNodeIndex(self, idx_node)) {
        char msg[150];
        sprintf(msg, "SumTree's node index %d is not allowed and must be between 0 and %d.", idx_node, self->total_nodes-1);

        PyErr_SetString(PyExc_ValueError, msg);
        return 0;
    }

    //Is node_idx root node?
    if (idx_node == 0)
        *idx_parent = -1;
    else
        *idx_parent = (idx_node + 1) / 2 - 1;

    return 1;
}

/* Get children indices of a node.
 * @param self a SumTree instance.
 * @param idx_node a node index.
 * @param idx_left_c left child index of idx_node.
 * @param idx_right_c right child index of idx_node.
 * @return 1 if this function does not raise any exception, 0 otherwise.*/
static int getChildrenIndices(SumTree* self, int idx_node, int* idx_left_c, int* idx_right_c) {
    if(!checkNodeIndex(self, idx_node)) {
        char msg[150];
        sprintf(msg, "SumTree's node index %d is not allowed and must be between 0 and %d.", idx_node, self->total_nodes-1);

        PyErr_SetString(PyExc_ValueError, msg);
        return 0;
    }

    //Get children indices.
    *idx_left_c = 2 * idx_node + 1;       // <--- 2 * (idx_node + 1) - 1
    *idx_right_c = 2 * idx_node + 2;      // <--- 2 * (idx_node + 1) + 1 - 1

    //Is idx_node a leaf node ?
    if ((int)pow(2, self->depth) - 1 <= idx_node && idx_node <= self->total_nodes - 1) {
        *idx_left_c = -1;
        *idx_right_c = -1;
    }

    return 1;
}

/* Sample ramdomly a transiction index.
 * @param self a SumTree instance.
 * @param idx_trans transiction index sampled.
 * @return 1 if this function does not raise any exception, 0 otherwise. */
static int getRandomTransiction(SumTree* self, int* idx_trans) {
    float* treeData = (float*) PyArray_DATA((PyArrayObject*) self->tree);
    float up = treeData[0];
    int idx_node = 0;
    float p = up * ((float)rand() / (float)RAND_MAX);
    
    int idx_lc, idx_rc;
    if(!getChildrenIndices(self, idx_node, &idx_lc, &idx_rc))
        return 0;

    while (idx_lc != -1 && idx_rc != -1) {
        if (p <= up - treeData[idx_rc]) {
            up -= treeData[idx_rc];
            idx_node = idx_lc;
        }
        else
            idx_node = idx_rc;

        if(!getChildrenIndices(self, idx_node, &idx_lc, &idx_rc))
            return 0;
    }

    *idx_trans = idx_node - ((int)pow(2, self->depth) - 1);
    return 1;
}

/* Return probability of a transiction index.
 * @param self a SumTree instance.
 * @param idx_trans a transiction index.
 * @param p probability of idx_trans.
 * @return 1 if this function does not raise any exception, 0 otherwise. */
static int getTransictionProbability(SumTree* self, int idx_trans, float* p) {
    if(idx_trans < 0 || idx_trans >= self->n_leaves) {
        char msg[150];
        sprintf(msg, "idx_trans is out of range for SumTree and is %d. It must be between 0 and %d.", idx_trans, self->n_leaves-1);

        PyErr_SetString(PyExc_IndexError, msg);
        return 0;
    }
    
    float* treeData = (float*) PyArray_DATA((PyArrayObject*) self->tree);

    if(treeData[0] == 0.f) {
        PyErr_SetString(PyExc_ZeroDivisionError, "SumTree's root node is 0.0");
        return 0;
    }

    *p = treeData[(int)pow(2, self->depth) - 1 + idx_trans] / treeData[0];
    return 1;
}


/* ======================================== 
 * ======== SUM TREE'S ATTRIBUTES =========
 * ======================================== */

static PyMemberDef SumTree_Attributes[] = {
    {"_n_leaves", T_INT, offsetof(SumTree, n_leaves), 0, "Number of leaves available of complete binary tree."},
    {"_depth", T_INT, offsetof(SumTree, depth), 0, "Depth of complete binary tree."},
    {"_total_nodes", T_INT, offsetof(SumTree, total_nodes), 0, "Total number nodes of complete binary tree."},
    {"_tree", T_OBJECT_EX, offsetof(SumTree, tree), 0, "Complete binary tree."},
    { NULL }
};


/* ======================================== 
 * ========== SUM TREE'S METHODS ==========
 * ======================================== */

/* ---------- METHOD __del__() ---------- */
static void SumTree_Dealloc(SumTree* self) {
    Py_XDECREF(self->tree);
    Py_TYPE(self)->tp_free((PyObject*) self);
}

/* ---------- METHOD __init__() ---------- */
static int SumTree_Init(SumTree* self, PyObject* args, PyObject* kwds) {
    //Retrieve parameters.
    int num_leaves = 0;
    if (!PyArg_ParseTuple(args, "i", &num_leaves))
        return -1;

    //Initialize SumTree's variables.
    self->n_leaves = num_leaves;
    self->depth = num_leaves % ((int) pow(2.0, (int) log2(num_leaves))) == 0 ? (int) log2(num_leaves) : ((int) log2(num_leaves)) + 1;
    self->total_nodes = ((int) pow(2.0, self->depth + 1)) - 1;
    
    //Initialize binary tree.
    npy_intp dimTree[1] = { self->total_nodes };
    self->tree = (PyObject*) PyArray_ZEROS(1, dimTree, NPY_FLOAT, 0);
    if (self->tree == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new binary tree.");
        return -1;
    }

    return 0;
}

/* ---------- METHOD set_priority() ---------- */
static PyObject* SumTree_SetPriority(SumTree* self, PyObject* args, void* closure) {    
    //Function's parameters.
    int idx;
    float prio;

    if (!PyArg_ParseTuple(args, "|if", &idx, &prio))
        return NULL;

    float* treeData = (float*) PyArray_DATA((PyArrayObject*) self->tree);

    //Set priority of a transiction.
    if(idx < 0 || idx >= self->n_leaves) {
        char msg[150];
        sprintf(msg, "idx is out of range for SumTree is %d. It must be between 0 and %d.", idx, self->n_leaves-1);

        PyErr_SetString(PyExc_IndexError, msg);
        return NULL;
    }
    treeData[(int)pow(2, self->depth) - 1 + idx] = prio;

    //Update cumulative priorities that have current transiction as leaf node.
    int idx_parent, idx_left_child, idx_right_child;

    if(!getParentIndex(self, (int)pow(2, self->depth) - 1 + idx, &idx_parent))
        return NULL;

    while (idx_parent != -1) {
        if(!getChildrenIndices(self, idx_parent, &idx_left_child, &idx_right_child))
            return NULL;

        treeData[idx_parent] = treeData[idx_left_child] + treeData[idx_right_child];

        if(!getParentIndex(self, idx_parent, &idx_parent))
            return NULL;
    }

    Py_INCREF(Py_None);
    return Py_None;
}

/* ---------- METHOD get_random_transiction() ---------- */
static PyObject* SumTree_GetRandomTransiction(SumTree* self, PyObject* Py_UNUSED(ignored)) {
    int idxTrans;
    if(!getRandomTransiction(self, &idxTrans))
        return NULL;

    return PyLong_FromLong(idxTrans);
}

/* ---------- METHOD sample_batch() ---------- */
static PyObject* SumTree_SampleBatch(SumTree* self, PyObject* args, void* closure) {
    //Method's parameter.
    int batchSize;
    if (!PyArg_ParseTuple(args, "i", &batchSize))
        return NULL;

    //Initialize an empty minibatch.
    npy_intp dims[1] = { batchSize };
    PyObject* batchIdxs = PyArray_SimpleNew(1, dims, NPY_INT);
    if (batchIdxs == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new batch indices.");
        return NULL;
    }

    int* batchIdxsData = (int*) PyArray_DATA((PyArrayObject*)batchIdxs);
    for (int i = 0; i < batchSize; i++)
        batchIdxsData[i] = -1;

    //Sample a minibatch.
    for (int i = 0; i < batchSize; i++) {
        int isDuplicated;
        int idxTrans;

        do {
            isDuplicated = 0;
            if(!getRandomTransiction(self, &idxTrans)) {
                Py_XDECREF(batchIdxs);
                batchIdxsData = NULL;
                return NULL;
            }

            //Check if idxTrans is already sampled.
            for (int j = 0; j < i && !isDuplicated; j++)
                if (idxTrans == batchIdxsData[j])
                    isDuplicated = 1;
        } 
        while (isDuplicated);

        //Store idxTrans into minibatch.
        batchIdxsData[i] = idxTrans;
    }

    return batchIdxs;
}

/* ---------- METHOD get_transiction_probability() ---------- */
PyObject* SumTree_GetTransictionProbability(SumTree* self, PyObject* args, void* closure) {
    //Method's parameter.
    int idx;
    if (!PyArg_ParseTuple(args, "i", &idx))
        return NULL;

    //Compute probability.
    float p;
    if(!getTransictionProbability(self, idx, &p))
        return NULL;

    return PyFloat_FromDouble(p);
}

/* ---------- METHOD get_batch_probability() ---------- */
PyObject* SumTree_GetBatchProbability(SumTree* self, PyObject* args, void* closure) {
    //Method's parameter.
    PyObject* batchIdxs = NULL;
    if (!PyArg_ParseTuple(args, "O", &batchIdxs))
        return NULL;

    Py_INCREF(batchIdxs);
   
    int* batchIdxsData = (int*) PyArray_DATA((PyArrayObject*) batchIdxs);
    npy_intp batchIdxsSize = PyArray_Size((PyArrayObject*) batchIdxs);

    //Create a ndarray to store probabilities.
    npy_intp probsDims[1] = { batchIdxsSize };
    PyObject* probs = PyArray_SimpleNew(1, probsDims, NPY_FLOAT);
    if (probs == NULL) {
        PyErr_SetString(PyExc_RuntimeError, "Failed to create new batch probabilities.");
        return NULL;
    }

    //Compute probabilties of batch.
    float* probsData = PyArray_DATA((PyArrayObject*) probs);
    for (int i = 0; i < batchIdxsSize; i++) {
        if(!getTransictionProbability(self, batchIdxsData[i], probsData + i)) {
            Py_XDECREF(probs);
            Py_XDECREF(batchIdxs);
            return NULL;
        }
    }
    
    Py_XDECREF(batchIdxs);
    return probs;
}

static PyMethodDef SumTree_Methods[] = {
    { "set_priority", (PyCFunction) SumTree_SetPriority, METH_VARARGS, "Set a transiction's priority on tree.\n\nParameters\n--------------------\nidx: int\n\tindex of a transiction\n\nprio : float\n\tpriority of the transiction\n"},
    { "get_random_transiction", (PyCFunction) SumTree_GetRandomTransiction, METH_NOARGS, "Return a transiction randomly.\n\nReturn\n--------------------\nidx_trans: int\n\tindex of transiction\n" },
    { "sample_batch", (PyCFunction) SumTree_SampleBatch, METH_VARARGS, "Sample a batch of transiction indices.\n\nParameter\n--------------------\nbatch_size: int\n\tbatch size\n\nReturn\n----------\nbatch_idxs: list\n\tbatch of transiction indices\n"},
    { "get_transiction_probability", (PyCFunction) SumTree_GetTransictionProbability, METH_VARARGS, "Return probability of a transiction.\n\nParameter\n--------------------\nidx : int\n\tindex of a transiction\n\nReturn\n--------------------\nprob : float\n\tprobability of the transiction\n"},
    { "get_batch_probability", (PyCFunction) SumTree_GetBatchProbability, METH_VARARGS, "Compute probabilities of transiction indices sampled.\n\nParameter\n----------\nbatch_idx: ndarray\n\tminibatch of transiction indices\n\nReturn\n----------\nprobs: ndarray\n\tprobabilities of transiction indices sampled\n" },
    { NULL }
};


/* ========================================
 * ============= SumTree TYPE =============
 * ======================================== */

static PyTypeObject SumTreeType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "sum_tree.SumTree",
    .tp_doc = PyDoc_STR("A complete binary tree that store cumulative priorities."),
    .tp_basicsize = sizeof(SumTree),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE,
    .tp_new = PyType_GenericNew,
    .tp_init = (initproc) SumTree_Init,
    .tp_dealloc = (destructor) SumTree_Dealloc,
    .tp_members = SumTree_Attributes,
    .tp_methods = SumTree_Methods
};


/* ======================================== 
 * ============ sum_tree MODULE ===========
 * ======================================== */

static PyModuleDef SumTreeModule = {
    PyModuleDef_HEAD_INIT,
    .m_name = "sum_tree",
    .m_doc = NULL,
    .m_size = -1,
};

/* ======================================== 
 * ============== Init Module =============
 * ======================================== */

PyMODINIT_FUNC PyInit_sum_tree() {
    //Import numpy
    import_array();
    
    //sum_tree module.
    PyObject* m = NULL;

    if (PyType_Ready(&SumTreeType) < 0)
        return NULL;

    m = PyModule_Create(&SumTreeModule);
    if (m == NULL)
        return NULL;

    Py_INCREF(&SumTreeType);
    if (PyModule_AddObject(m, "SumTree", (PyObject*)&SumTreeType) < 0) {
        Py_DECREF(&SumTreeType);
        Py_DECREF(m);

        return NULL;
    }

    return m;
}