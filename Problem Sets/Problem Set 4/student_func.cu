//Udacity HW 4
//Radix Sorting

#include "utils.h"
#include <thrust/host_vector.h>

/* Red Eye Removal
   ===============

   For this assignment we are implementing red eye removal.  This is
   accomplished by first creating a score for every pixel that tells us how
   likely it is to be a red eye pixel.  We have already done this for you - you
   are receiving the scores and need to sort them in ascending order so that we
   know which pixels to alter to remove the red eye.

   Note: ascending order == smallest to largest

   Each score is associated with a position, when you sort the scores, you must
   also move the positions accordingly.

   Implementing Parallel Radix Sort with CUDA
   ==========================================

   The basic idea is to construct a histogram on each pass of how many of each
   "digit" there are.   Then we scan this histogram so that we know where to put
   the output of each digit.  For example, the first 1 must come after all the
   0s so we have to know how many 0s there are to be able to start moving 1s
   into the correct position.

   1) Histogram of the number of occurrences of each digit
   2) Exclusive Prefix Sum of Histogram
   3) Determine relative offset of each digit
        For example [0 0 1 1 0 0 1]
                ->  [0 1 0 1 2 3 2]
   4) Combine the results of steps 2 & 3 to determine the final
      output location for each element and move it there

   LSB Radix sort is an out-of-place sort and you will need to ping-pong values
   between the input and output buffers we have provided.  Make sure the final
   sorted results end up in the output buffer!  Hint: You may need to do a copy
   at the end.

 */
//=========================================================
//              Device functions
//=========================================================
__global__
void histogram_kernel(unsigned int pass,
                      unsigned int * d_bins,
                      unsigned int* const d_input,
                      const int size){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id >= size){
    return;
  }

}

__global__
void scan_kernel(unsigned int pass,
                    unsigned int const * d_inputVals,
                    unsigned int * d_output,
                    const int size,
                    unsigned int base,
                    unsigned int threadSize){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id >= size){
    return;
  }

}

__global__
void move_kernel(unsigned int pass,
    unsigned int* const d_inputVals,
    unsigned int* const d_inputPos,
    unsigned int* d_outputVals,
    unsigned int* d_outputPos,
    unsigned int* d_outputMove,
    unsigned int* const d_scanned,
    unsigned int  one_pos,
    const size_t numElems){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id >= size){
    return;
  }
}
//=========================================================
//              Host functions
//=========================================================
int get_max_size(int problemSize, int blockSize){
  return (int)ceil((float)problemSize / (float)blockSize) + 1;
}
void your_sort(unsigned int* const d_inputVals,
               unsigned int* const d_inputPos,
               unsigned int* const d_outputVals,
               unsigned int* const d_outputPos,
               const size_t numElems)
{
  //TODO
  //PUT YOUR SORT HERE
  dim3 blockSize(1024);
  dim3 gridSize(get_max_size(numElems, blockSize.x));
}
