#include <cuda_runtime.h>
#include <stdio.h>
#include <sys/time.h>

#ifndef __COMMON_H
#define __COMMON_H

/* Device code utils */
#define CHECK(call) \
{ \
  const cudaError_t error = call; \
  if (error != cudaSuccess) \
  { \
    printf("Error: %s:%d, ", __FILE__, __LINE__); \
    printf("code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
    exit(1); \
  } \
}



/*  Host code utils */
void checkResult(const char *types, void *A, void *B, const int N) {
  bool match = 1;
  int i;

  if( ! strcmp(types, "float") ){ 
    double epsilon = 1.0E-8;
    float *hostRef = (float *)A;
    float *gpuRef = (float *)B;

    for (i=0; i<N; i++) {
      if (abs( hostRef[i] - gpuRef[i]) > epsilon) {
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
        break;
      }
    }
  } else if ( ! strcmp(types, "int") ){
    int *hostRef = (int *)A;
    int *gpuRef = (int *)B;
    for (i=0; i<N; i++) {
      if ((hostRef[i] - gpuRef[i])){
        match = 0;
        printf("Arrays do not match!\n");
        printf("host %5.2f gpu %5.2f at current %d\n",hostRef[i],gpuRef[i],i);
        break;
      }
    }
  } else {
    match = 0;
    fprintf(stderr, "[x] Types do not match!\n");
  }

  if (match) 
    printf("Arrays match.\n\n");
}


int initialData(const char *type, void *A,int size) {
  // generate different seed for random number
  time_t t;
  srand((unsigned) time(&t));

  if (!strcmp(type, "float")) {
    float *ip = (float *) A;
    for (int i=0; i<size; i++) {
      ip[i] = (float)( rand() & 0xFF )/10.0f;
    }
  } else if (!strcmp(type, "int")) {
    int *ip = (int *) A;
    for (int i=0; i<size; i++) {
      ip[i] = ( rand() & 0xFF ) % 10 ;
    }
  } else {
    fprintf(stderr, "[x] Types do not match!\n");
    return 1;
  }
  return 0;
}


void printMatrix( int *C, const int nx, const int ny) {
  int *ic = C;
  printf("\nMatrix: (%d.%d)\n",nx,ny);

  for (int iy=0; iy<ny; iy++) {
    for (int ix=0; ix<nx; ix++) {
      printf("%3d",ic[ix]);
    }
    ic += nx;
    printf("\n");
  }
  printf("\n");
}

inline double seconds()
{
  struct timeval tp;
  struct timezone tzp;
  int i = gettimeofday(&tp, &tzp);
  return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

int cudaSupport(void) {
  int deviceCount = 0;
  cudaError_t error_id; 
  if (( error_id = cudaGetDeviceCount(&deviceCount))!= cudaSuccess) {
    fprintf(stderr, "cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
    exit(EXIT_FAILURE);
  } 
  return deviceCount;
}

void printCudaDesc(int dev){
  int deviceCount = 0;
  int driverVersion = 0, runtimeVersion = 0;
  cudaDeviceProp deviceProp;

  if (!(deviceCount=cudaSupport())) {
    fprintf(stderr, "[x] Cuda not supported!\n");
    exit(EXIT_FAILURE);
  }

  if (deviceCount < (dev - 1)) {
    fprintf(stderr, "[x] Cuda deviceID not present..\n");
    exit(EXIT_FAILURE);
  }

  cudaSetDevice(dev);
  cudaGetDeviceProperties(&deviceProp, dev);
  printf("\n\nDevice %d: \"%s\"\n", dev, deviceProp.name);
  cudaDriverGetVersion(&driverVersion);
  cudaRuntimeGetVersion(&runtimeVersion);
  printf(" CUDA Driver Version / Runtime Version %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
  printf(" CUDA Capability Major/Minor version number: %d.%d\n", deviceProp.major, deviceProp.minor);
  printf(" ----------------------------------------------------------\n");
  printf(" Number of multiprocessors: %d\n", deviceProp.multiProcessorCount);
  printf(" Total amount of global memory: %.2f MBytes (%llu bytes)\n", (float)deviceProp.totalGlobalMem/(pow(1024.0,3)), (unsigned long long) deviceProp.totalGlobalMem);
  printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);
  printf(" Memory Clock rate: %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
  printf(" Memory Bus Width: %d-bit\n", deviceProp.memoryBusWidth);
  if (deviceProp.l2CacheSize) {
    printf(" L2 Cache Size: %d bytes\n", deviceProp.l2CacheSize);
  }
  printf(" ----------------------------------------------------------\n");
  printf(" Max Texture Dimension Size (x,y,z) 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n", 
           deviceProp.maxTexture1D, deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1], deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
  printf(" Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
           deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1], deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

  printf(" ----------------------------------------------------------\n");
  printf(" Total amount of constant memory: %4.2f KB\n", deviceProp.totalConstMem / 1024.0 );
  printf(" Total amount of shared memory per block: %4.2f KB\n", deviceProp.sharedMemPerBlock / 1024.0);
  printf(" Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
  printf(" Warp size: %d\n", deviceProp.warpSize);
  printf(" Maximum number of threads per block: %d\n", deviceProp.maxThreadsPerBlock);
  printf(" Maximum number of threads per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
  printf(" Maximum number of warps per multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor/32);
  printf(" ----------------------------------------------------------\n");
  printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
  printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
  printf(" Maximum memory pitch: %lu bytes\n", deviceProp.  memPitch);
  printf("\n\n");
}

__global__ void warmup(int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0;

    if ((tid / warpSize) % 2 == 0)
    {
        ia = 100;
    }
    else
    {
        ib = 200;
    }

    c[tid] = ia + ib;
}


#endif

