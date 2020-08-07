#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cimage/cimage.h"

/*
    Il file seguente serve a testare sia l'implementazione della bitmap
    che la conversione RGB-LAB.
    Sono presenti inesattezze che però sono state corrette nella versione CUDA.

64 bit:
rm -rf main && gcc -D DEBUG -fpic -O0 -std=c99 -Wall test_bmp.c cimage/imagebmp.c -o main -lm && ./main

32 bit:
rm -rf main && gcc -D DEBUG -D X86_32 -fpic -O0 -std=c99 -Wall test_bmp.c cimage/imagebmp.c -o main -lm -m32 && ./main
*/

static const struct img_ops bmp_op = {
    .read = &readBMP,
    .write = &writeBMP
};

#define __init_img( oops) {       \
    .f_op = oops,                 \
    .flags = 0xfbadfbad,          \
    .hPixels = 0,                 \
    .wPixels = 0,                 \
    .wBytes  = 0                  \
}


#define New_bmp(name) \
  struct imgProp name = __init_img( &bmp_op)


                                            //     r           g          b
const static double matr_xyz2rgb[3][3] =  {
                                                  {3.2404542, -0.9692660, 0.0556434},   // x
                                                  {-1.5371385, 1.8760108, -0.2040259},  // y
                                                  {-0.4985314, 0.0415560, 1.0572252}    // z
                                          };


                                            //   X           Y           Z
const static double matr_rgb2xyz[3][3] =  {
                                                {0.4124564, 0.2126729, 0.0193339}, // R
                                                {0.3575761, 0.7151522, 0.1191920}, // G
                                                {0.1804375, 0.0721750, 0.9503041}, // B
                                          };




const static double delta = 0.20689655172413793;
const static double Xn=95.047, Yn=100.0, Zn=108.883;

/* Conversion from XYZ to RGB */
void linearRGB(double *xyz, double *ret) {
  for (int i=0; i<3; i++)
    ret[i] = matr_xyz2rgb[0][i] * xyz[0] + matr_xyz2rgb[1][i] * xyz[1] + matr_xyz2rgb[2][i] * xyz[2];
}

void gammaCorrection(double *rgb_double, pxl *ret) {  // Ok
  for (int i=0; i<3; i++) {
    if ( rgb_double[i] <= 0.0031308)
      ret[i] =  round(rgb_double[i] * 12.92 * 255.);
    else
      ret[i] = round( ((1.055 * pow(rgb_double[i], 1. / 2.4)) - 0.055) * 255.);
  }
}


/* Conversion from RGB to XYZ */
void gammaCorrection_inv(pxl *rgb, double *ret) {   // Ok
  double tmp[3] = { rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0};
  for (int i=0; i<3; i++) {
    if (tmp[i] <= 0.04045)
      ret[i] = tmp[i] / 12.92;
    else
      ret[i] = pow( (tmp[i] + 0.055) / 1.055, 2.4);
  }
}

void XYZlinear(double *rgb_double, double *ret) {
  for (int i=0; i<3; i++) 
    ret[i] = matr_rgb2xyz[0][i] * rgb_double[0] + matr_rgb2xyz[1][i] * rgb_double[1] + matr_rgb2xyz[2][i] * rgb_double[2];
}



int main(int argc, char *argv[], char *envp[]) {
  New_bmp(ip);
  //struct imgProp *ip = malloc(sizeof(struct imgProp));
  ip.f_op = &bmp_op;

  char filename[] = "lenna.bmp";
  char out_filename[] = "out_lenna.bmp";

  pxl *output_rgb = ip.f_op->read(&ip, filename);
  unsigned int nElem = ip.hPixels * ip.wPixels;

  /* memcpy to compare here .. */
  double *output_tmp = malloc(sizeof(double) * nElem * 3);
  double *output_xyz = malloc(sizeof(double) * nElem * 3);
  double *output_lab = malloc(sizeof(double) * nElem * 3);
 
  /* Convert from RGB to XYZ */ 
  for (unsigned int i=0; i< nElem; i++) {
    gammaCorrection_inv(&output_rgb[i*3], &output_tmp[i*3]);
    XYZlinear(&output_tmp[i*3], &output_xyz[i*3]);
  }
  
  /* Convert from XYZ to LAB */
  for (unsigned int i=0; i< nElem; i++) {
    double x = output_xyz[i*3] / Xn;
    double y = output_xyz[i*3 + 1] / Yn;
    double z = output_xyz[i*3 + 2] / Zn;
    
    double y_p = pow(y, 1.0/3.0);

    x = x > 0.008856 ? pow(x, 1.0/3.0) : 7.787 * x + 0.13793;
    y = y > 0.008856 ? pow(y, 1.0/3.0) : 7.787 * y + 0.13793;
    z = z > 0.008856 ? pow(z, 1.0/3.0) : 7.787 * z + 0.13793;

    double L = y > 0.008856 ? (116.0 * y_p - 16.0) : 903.3 * y;
    double A = (x - y) * 500.0;
    double B = (y - z) * 200.0;

    output_lab[i*3] = L;
    output_lab[i*3 + 1] = A;
    output_lab[i*3 + 2] = B;
  }
  
  /* Convert from LAB to XYZ */
  
  for (unsigned int i=0; i< nElem; i++) {
    double L = output_lab[i*3];
    double A = output_lab[i*3 + 1];
    double B = output_lab[i*3 + 2];
    double P = (L + 16.) / 116. ;
    
    double x = Xn * pow(P + A / 500., 3.0);
    double y = Yn * pow(P, 3.0);
    double z = Zn * pow(P - B / 200., 3.0);

    output_xyz[i*3] = x;
    output_xyz[i*3 + 1] = y;
    output_xyz[i*3 + 2] = z;
  }
  
  /* Convert from XYZ to RGB */
  for (unsigned int i=0; i< nElem; i++) {
    linearRGB(&output_xyz[i*3], &output_tmp[i*3]);
    gammaCorrection(&output_tmp[i*3], &output_rgb[i*3]);
  }
  
 
  /* Write it on the bitmap file */ 
  ip.f_op->write(&ip, output_rgb, out_filename);

  return 0;
}



