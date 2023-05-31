//first make clear, here we initialize the MY_PyArray_API
#define INIT_NUMPY_ARRAY_CPP

//now include the arrayobject.h, which defines
//void **MyPyArray_API
#include "use_numpy.h"

//now the old trick with initialization:
int init_numpy() {
    import_array() // PyError if not successful
    return 0;
}

//const static int numpy_initialized =  init_numpy();
