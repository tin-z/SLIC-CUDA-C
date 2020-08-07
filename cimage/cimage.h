#ifndef __img_1337
#define __img_1337


typedef unsigned char pxl;    // pixel element


struct imgProp {
  const struct img_ops * f_op;
  unsigned int flags;      // future usage
  int hPixels;
  int wPixels;
  unsigned char hdr[54];
  unsigned long int wBytes;
};


struct Pixel {
	pxl R;
	pxl G;
	pxl B;
};


struct img_ops {
  pxl* (*read) (struct imgProp *, char *);
  int (*write) (struct imgProp *, pxl*, char *);
};


pxl* readBMP(struct imgProp * ip, char* filename);         // Load a BMP image
int writeBMP(struct imgProp * ip, pxl* img, char* filename);  // Store a BMP image


#endif
