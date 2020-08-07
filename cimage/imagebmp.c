#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#include "../common/printutil.h"
#include "cimage.h"
#define SIZE 256



extern long my_memset(char *buff, char a, unsigned int size); /* Prototype */
extern long my_memmov(unsigned char *dest, unsigned char *from, int size);


#ifdef X86_32
__asm__(
  "my_memset:\n"
  "mov 12(%esp), %ecx\n"
  "mov 8(%esp), %eax\n"
  "mov 4(%esp), %edi\n"
  "rep stosb\n"
  "ret\n"
);

__asm__(
  "my_memmov:\n"
  "mov 12(%esp), %ecx\n"
  "mov 8(%esp), %esi\n"
  "mov 4(%esp), %edi\n"
  "rep movsb\n"
  "ret\n"
);


#else
__asm__(
  "my_memset:\n"
  "mov %rsi, %rax\n"
  "mov %rdx, %rcx\n"
  "rep stosb\n"
  "ret\n"
);

__asm__(
  "my_memmov:\n"
  "mov %rdx, %rcx\n"
  "rep movsb\n"
  "ret\n"
);


#endif



// Read a BMP image
pxl* readBMP(struct imgProp * ip, char* filename) {

  unsigned int i;
  FILE* fp = fopen(filename, "rb");
  char buff[SIZE];
  if (fp == NULL) {
    my_memset(buff, 0, SIZE);
    snprintf(buff, SIZE, "%s Not Found", filename);
    printErr(buff);
  }

  pxl HeaderInfo[54];
  fread(HeaderInfo, sizeof(pxl), 54, fp);   // read the 54-byte header

  // extract image height and width from header
  int width = *(int*)(HeaderInfo + 18);
  int height = *(int*)(HeaderInfo + 22);

  my_memmov(ip->hdr, HeaderInfo, 54);
  
  ip->hPixels = height;
  ip->wPixels = width;
  int RowBytes = (width * 3 + 3) & (~3);

  ip->wBytes = RowBytes;

#ifdef DEBUG    
  my_memset(buff, 0, SIZE);
  snprintf(buff, SIZE, "Input BMP File name: %20s  (%u x %u)", filename, ip->hPixels, ip->wPixels);
  printG(buff);
#endif

  unsigned int  nElem = height * RowBytes;
  pxl *output = malloc(nElem * sizeof(pxl));

  for (i = 0; i < nElem; i+= RowBytes)
    fread((output + i), sizeof(pxl), RowBytes, fp);

  fclose(fp);
  return output;  // remember to free() it in caller!
}


// Store a BMP image
int writeBMP(struct imgProp* ip, pxl* img, char* filename) {
	FILE* fp = fopen(filename, "wb");
  char buff[SIZE];
  unsigned int i;
	if (fp == NULL) {
    my_memset(buff, 0, SIZE);
    snprintf(buff, SIZE, "File creation error: %s", filename);
    printErr(buff);
	}

	//write header
  fwrite(ip->hdr, sizeof(pxl), 54, fp);   // read the 54-byte header

	//write data
  unsigned int RowBytes = ip->wBytes;
  unsigned int nElem = ip->hPixels * RowBytes;
  for (i = 0; i < nElem; i+= RowBytes)
    fwrite((img + i), sizeof(pxl), RowBytes, fp);

#ifdef DEBUG    
  my_memset(buff, 0, SIZE);
  snprintf(buff, SIZE, "Output BMP File name: %20s  (%u x %u)", filename, ip->hPixels, ip->wPixels);
  printG(buff);
#endif

	return fclose(fp);
}

