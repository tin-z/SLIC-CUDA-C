#include "../converter.h"
#include "../common/printutil.h"
#include "../common/common.h"
#include <stdio.h>
#include <stdlib.h>

/* To compile this program you need to launch nvcc as follows:
Optimized version: nvcc -D DEBUG -O3 -arch=sm_20 test1.cu ../cimage/imagebmp.c ../converter.cu -o test1
Not-optimized ver: nvcc -D DEBUG -D NO_OPTIMIZED_RGB2LAB -O3 -arch=sm_20 test1.cu ../cimage/imagebmp.c ../converter.cu -o test1
*/


/////////////////  UTILS variables, structs and imports

#ifdef DEBUG
int debug = 1;
#else
int debug = 0;
#endif

#ifdef SNAPSHOT
int snapshot = 1;
#else
int snapshot = 0;
#endif


#define SIZE 256

typedef unsigned char pxl;    // pixel element

struct imgProp {
  const struct img_ops * f_op;
  unsigned int flags;      // future usage
  int hPixels;
  int wPixels;
  unsigned char hdr[54];
  unsigned long int wBytes;
};

extern "C" {
  pxl* readBMP(struct imgProp * ip, char* filename);
  int writeBMP(struct imgProp * ip, pxl* img, char* filename);
}



/////////////////  Main
int main(int argc, char **argv) {

  if(argc < 2 ){
    fprintf(stderr, "Usage: %s <filename.bmp>\n", argv[0]);
    fprintf(stderr, "  as output you will have 'out_<filename.bmp>'\n");
    return 1;
  }

  struct imgProp ip;

  char *filename = argv[1];
  char buff[SIZE];

  pxl *output_rgb = readBMP(&ip, filename);
  unsigned int nRow = ip.hPixels, nCol = ip.wBytes / 3;
  unsigned int nElem = nRow * nCol;
  unsigned int nBytes = nElem * sizeof(float) * 3;

  /* Glue CODE, converti SOA in SoA, più avanti scriverlo in CUDA */
  float * output = (float *)malloc(nBytes);
  for (int i=0; i<nElem; i++) {
    *(output + i) = (float)((unsigned int)output_rgb[i*3]);
    *(output + nElem + i) = (float)((unsigned int)output_rgb[i*3 + 1]);
    *(output + 2*nElem + i) = (float)((unsigned int)output_rgb[i*3 + 2]);
  }

  float *d_pxl_rgb, *d_pxl_lab;
  CHECK(cudaMalloc((float **)&d_pxl_rgb, nBytes ));
  CHECK(cudaMalloc((float **)&d_pxl_lab, 4 * nBytes / 3));
  CHECK(cudaMemcpy(d_pxl_rgb, output, nBytes, cudaMemcpyHostToDevice));

  int dimx = 16, dimy = 16;
  dim3 block (dimx, dimy);
  dim3 grid ((nCol + block.x - 1) / block.x, (nRow + block.y - 1) / block.y);
  convertRGB2Lab<<<grid, block>>>(d_pxl_rgb, d_pxl_lab, nCol, nRow);
  convertLab2RGB<<<grid, block>>>(d_pxl_lab, d_pxl_rgb, nCol, nRow);
  CHECK(cudaMemcpy(output, d_pxl_rgb, nBytes, cudaMemcpyDeviceToHost));

  /* Glue CODE, converti SOA in SoA, più avanti scriverlo in CUDA */
  for (int i=0; i<nElem; i++) {
    output_rgb[i*3] = (pxl)((unsigned int)*(output + i));
    output_rgb[i*3 + 1] = (pxl)((unsigned int)*(output + nElem + i));
    output_rgb[i*3 + 2] = (pxl)((unsigned int)*(output + 2*nElem + i));
  }

  memset(buff, 0, SIZE);
  snprintf(buff, SIZE, "out_%s", filename);

  // writeBMP(&ip, output_rgb, buff);

  CHECK(cudaFree(d_pxl_rgb));
  CHECK(cudaFree(d_pxl_lab));

  free(output);
  free(output_rgb);

  // reset device
  cudaDeviceReset();

  return 0;
}

