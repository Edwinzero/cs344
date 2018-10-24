/* Udacity Homework 3
   HDR Tone-mapping

  Background HDR
  ==============

  A High Dynamic Range (HDR) image contains a wider variation of intensity
  and color than is allowed by the RGB format with 1 byte per channel that we
  have used in the previous assignment.

  To store this extra information we use single precision floating point for
  each channel.  This allows for an extremely wide range of intensity values.

  In the image for this assignment, the inside of church with light coming in
  through stained glass windows, the raw input floating point values for the
  channels range from 0 to 275.  But the mean is .41 and 98% of the values are
  less than 3!  This means that certain areas (the windows) are extremely bright
  compared to everywhere else.  If we linearly map this [0-275] range into the
  [0-255] range that we have been using then most values will be mapped to zero!
  The only thing we will be able to see are the very brightest areas - the
  windows - everything else will appear pitch black.

  The problem is that although we have cameras capable of recording the wide
  range of intensity that exists in the real world our monitors are not capable
  of displaying them.  Our eyes are also quite capable of observing a much wider
  range of intensities than our image formats / monitors are capable of
  displaying.

  Tone-mapping is a process that transforms the intensities in the image so that
  the brightest values aren't nearly so far away from the mean.  That way when
  we transform the values into [0-255] we can actually see the entire image.
  There are many ways to perform this process and it is as much an art as a
  science - there is no single "right" answer.  In this homework we will
  implement one possible technique.

  Background Chrominance-Luminance
  ================================

  The RGB space that we have been using to represent images can be thought of as
  one possible set of axes spanning a three dimensional space of color.  We
  sometimes choose other axes to represent this space because they make certain
  operations more convenient.

  Another possible way of representing a color image is to separate the color
  information (chromaticity) from the brightness information.  There are
  multiple different methods for doing this - a common one during the analog
  television days was known as Chrominance-Luminance or YUV.

  We choose to represent the image in this way so that we can remap only the
  intensity channel and then recombine the new intensity values with the color
  information to form the final image.

  Old TV signals used to be transmitted in this way so that black & white
  televisions could display the luminance channel while color televisions would
  display all three of the channels.


  Tone-mapping
  ============

  In this assignment we are going to transform the luminance channel (actually
  the log of the luminance, but this is unimportant for the parts of the
  algorithm that you will be implementing) by compressing its range to [0, 1].
  To do this we need the cumulative distribution of the luminance values.

  Example
  -------

  input : [2 4 3 3 1 7 4 5 7 0 9 4 3 2]
  min / max / range: 0 / 9 / 9

  histo with 3 bins: [4 7 3]

  cdf : [4 11 14]


  Your task is to calculate this cumulative distribution by following these
  steps.

*/
#include <limits.h>
#include <float.h>
#include <math.h>
#include <stdio.h>
#include "utils.h"
//=====================================================
//                  Kernels
//=====================================================
__global__
void reduce_minmax(const float *d_in, float *d_out, const size_t size, const int isMin){
  extern __shared__ float shared[];
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  float THRES = FLT_MAX;
  if(!isMin){
    THRES = -FLT_MAX;
  }
  if(gid < size){
    shared[tid] = d_in[gid];  // if size is solvable, then copy to shared mem
  }
  else{
      shared[tid] = THRES;
  }

  // sync
  __syncthreads();

  if(gid >= size){
    if(tid == 0){
        d_out[blockIdx.x] = THRES;
    }
    return;
  }

  for(int i = blockDim.x * 0.5f; i > 0; i *= 0.5f){
    if(tid < i){
      float v = 0.0f;
      isMin ? v = min(shared[tid], shared[tid + i]) : v = max(shared[tid], shared[tid + i]);
      shared[tid] = v;
    }
  }

  if(tid == 0){
    d_out[blockIdx.x] = shared[0];
  }
}
__global__
void reduce_min(const float *d_in, float *d_out, const size_t size){
  extern __shared__ float shared[];
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  const float THRES = FLT_MAX;
  if(gid < size){
    shared[tid] = d_in[gid];  // if size is solvable, then copy to shared mem
  }
  else{
      shared[tid] = THRES;
  }

  // sync
  __syncthreads();

  if(gid >= size){
    if(tid == 0){
        d_out[blockIdx.x] = THRES;
    }
    return;
  }

  for(int i = blockDim.x * 0.5f; i > 0; i *= 0.5f){
    if(tid < i){
      shared[tid] = min(shared[tid], shared[tid + i]);
    }
  }

  if(tid == 0){
    d_out[blockIdx.x] = shared[0];
  }
}
__global__
void reduce_max(const float *d_in, float *d_out, const size_t size){
  extern __shared__ float shared[];
  int gid = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  const float THRES = -FLT_MAX;
  if(gid < size){
    shared[tid] = d_in[gid];  // if size is solvable, then copy to shared mem
  }
  else{
      shared[tid] = THRES;
  }

  // sync
  __syncthreads();

  if(gid >= size){
    if(tid == 0){
        d_out[blockIdx.x] = THRES;
    }
    return;
  }

  for(int i = blockDim.x * 0.5f; i > 0; i *= 0.5f){
    if(tid < i){
      shared[tid] = max(shared[tid], shared[tid + i]);
    }
  }

  if(tid == 0){
    d_out[blockIdx.x] = shared[0];
  }
}
__global__
void histogram_kernel(unsigned int* d_bins, const float* d_logLuminance, const size_t size, const float lmin, const float lmax, const int numBins){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id >= size){
    return;
  }
  float range = lmax - lmin;
  int bin = (int)(((d_logLuminance[id]-lmin)/range)*numBins);

  atomicAdd(&d_bins[bin], 1);

}
__global__
void scan_kernel(unsigned int* d_bins, int size){
  int id = threadIdx.x + blockDim.x * blockIdx.x;
  if(id >= size){
    return;
  }

  for(int i = 1; i < size; i *= 2){
    int ss = id -i;
    unsigned int val = 0;
    if(ss > 0){
      val = d_bins[ss];
    }
    __syncthreads();
    if(ss >= 0){
      d_bins[id] += val;
    }
  }
}

//=====================================================
//                  Host
//=====================================================

int get_max_size(int n, int d) {
    return (int)ceil( (float)n/(float)d ) + 1;
}
float find_min_max(const float* const d_logLuminance, const size_t size, int numRows, int numCols, int isMin){
    float *d_in;
    checkCudaErrors(cudaMalloc(&d_in, sizeof(float) * size));
    cudaMemcpy(d_in, d_logLuminance, sizeof(float) * size, cudaMemcpyDeviceToDevice);

    float *d_out;
    const int blockWidth = 32;
    const dim3 blockSize(blockWidth);
    const dim3 gridSize(get_max_size(size, blockSize.x));
    const int shared_mem_size = sizeof(float) * blockWidth;

    size_t cur_size = size;
    while(1){
      //student_func.cu:254:0: error: unterminated argument list invoking macro "checkCudaErrors"
      // this error indicates you have error on parenthesis
      checkCudaErrors(cudaMalloc(&d_out, sizeof(float) * get_max_size(cur_size, blockWidth)));
      //if(isMin){
      //  reduce_min<<<gridSize, blockSize, shared_mem_size>>>(d_in, d_out, cur_size);
      //}else{
      //  reduce_max<<<gridSize, blockSize, shared_mem_size>>>(d_in, d_out, cur_size);
      //}
      reduce_minmax<<<gridSize, blockSize, shared_mem_size>>>(d_in, d_out, cur_size, isMin);
      cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

      checkCudaErrors(cudaFree(d_in));
      d_in = d_out;
      if(cur_size < blockSize.x){
        break;
      }

      cur_size = get_max_size(cur_size, blockWidth);
    }

    float h_out = 0.0f;
    cudaMemcpy(&h_out, d_out, sizeof(float), cudaMemcpyDeviceToHost);
    checkCudaErrors(cudaFree(d_out));
    return h_out;
}
void your_histogram_and_prefixsum(const float* const d_logLuminance,
                                  unsigned int* const d_cdf,
                                  float &min_logLum,
                                  float &max_logLum,
                                  const size_t numRows,
                                  const size_t numCols,
                                  const size_t numBins)
{
  //TODO
  /*Here are the steps you need to implement
    1) find the minimum and maximum value in the input logLuminance channel
       store in min_logLum and max_logLum
    2) subtract them to find the range
    3) generate a histogram of all the values in the logLuminance channel using
       the formula: bin = (lum[i] - lumMin) / lumRange * numBins
    4) Perform an exclusive scan (prefix sum) on the histogram to get
       the cumulative distribution of luminance values (this should go in the
       incoming d_cdf pointer which already has been allocated for you)       */


       // 1. find min and max value in min_logLum, max_logLum by reduce
       size_t imgsize = numRows * numCols;
       min_logLum = find_min_max(d_logLuminance, imgsize, numRows, numCols, 1);
       max_logLum = find_min_max(d_logLuminance, imgsize, numRows, numCols, 0);

       printf("got min of %f\n", min_logLum);
       printf("got max of %f\n", max_logLum);
       printf("numBins %d\n", numBins);
       // 2. find range
       // 3. generate histogram with bin size_t
       unsigned int *d_bins;
       size_t hist_size = sizeof(unsigned int)*numBins;

       checkCudaErrors(cudaMalloc(&d_bins, hist_size));
       checkCudaErrors(cudaMemset(d_bins, 0, hist_size)); // (dest, value, size)
       dim3 blockSize(1024);    // only blockSize 1024 works
       dim3 gridSize(get_max_size(imgsize, blockSize.x));
       histogram_kernel<<< gridSize, blockSize >>>(d_bins, d_logLuminance, imgsize, min_logLum, max_logLum, numBins);
       cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());

       // 4. perform exclusive scan on histogram
       dim3 scanGridSize(get_max_size(numBins, blockSize.x));

       scan_kernel<<< scanGridSize, blockSize >>>(d_bins, numBins);
       cudaDeviceSynchronize(); checkCudaErrors(cudaGetLastError());
       cudaMemcpy(d_cdf, d_bins, hist_size, cudaMemcpyDeviceToDevice);
       checkCudaErrors(cudaFree(d_bins));
}
