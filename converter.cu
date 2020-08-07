#include "converter.h"

// use CIE Lab chromaticity coordinates
#define nCIE_LAB_D65_xn 0.950455F
#define nCIE_LAB_D65_yn 1.0F
#define nCIE_LAB_D65_zn 1.088753F


#ifdef NO_OPTIMIZED_RGB2LAB


__global__ void convertRGB2Lab(float *pxl_rgb, float *pxl_lab_label, int nCol, int nRow, int N)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  /* Convert from RGB to XYZ */
  float b = pxl_rgb[idx];
  float g = pxl_rgb[idx + N];
  float r = pxl_rgb[idx + 2*N];

  unsigned char p = b <= 0.04045;
  unsigned char p2 = g <= 0.04045;
  unsigned char p3 = b <= 0.04045;
  b = p ? b / 12.92 : powf( (b + 0.055) / 1.055, 2.4);
  g = p2 ? g / 12.92 : powf( (g + 0.055) / 1.055, 2.4);
  r = p3 ? r / 12.92 : powf( (r + 0.055) / 1.055, 2.4);

  /* Convert from XYZ to LAB */
  float Xn=95.047, Yn=100.0, Zn=108.883;
  float x = (r * 0.412453 + g * 0.357580 + b * 0.180423) / Xn;
  float y = (r * 0.212671 + g * 0.715160 + b * 0.072169) / Yn;
  float z = (r * 0.019334 + g * 0.119193 + b * 0.950227) / Zn;

  float y_p = powf(y, 1.0/3.0);

  x = x > 0.008856 ? powf(x, 1.0/3.0) : 7.787 * x + 0.13793;
  y = y > 0.008856 ? powf(y, 1.0/3.0) : 7.787 * y + 0.13793;
  z = z > 0.008856 ? powf(z, 1.0/3.0) : 7.787 * z + 0.13793;

  float L = y > 0.008856 ? (116.0 * y_p - 16.0) : 903.3 * y;
  float A = (x - y) * 500.0;
  float B = (y - z) * 200.0;

  __syncthreads();

  pxl_lab_label[idx] = L;
  pxl_lab_label[idx + N] = A;
  pxl_lab_label[idx + 2*N] = B;
  pxl_lab_label[idx + 3*N] = -1;  // for label centroid

}


__global__ void convertLab2RGB(float *pxl_lab_label, float *pxl_rgb, int nCol, int nRow, int N)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  /* Convert from LAB to XYZ */
  float L = pxl_lab_label[idx];
  float A = pxl_lab_label[idx + N];
  float B = pxl_lab_label[idx + 2*N];
  float P = (L + 16.) / 116. ;

  float Xn=95.047, Yn=100.0, Zn=108.883;

  double x = Xn * powf(P + A / 500., 3.0);
  double y = Yn * powf(P, 3.0);
  double z = Zn * powf(P - B / 200., 3.0);


  /* Convert from XYZ to RGB */
  float r = x * 3.2404542 + y * -1.5371385 + z * -0.4985314;
  float g = x * -0.9692660 + y * 1.8760108 + z * 0.0415560;
  float b = x * 0.0556434 + y * -0.2040259 + z * 1.0572252;

  r = r <= 0.0031308 ?  r * 12.92 * 255. : ((1.055 * powf(r, 0.4166666666666667)) - 0.055) * 255.;
  g = g <= 0.0031308 ?  g * 12.92 * 255. : ((1.055 * powf(g, 0.4166666666666667)) - 0.055) * 255.;
  b = b <= 0.0031308 ?  b * 12.92 * 255. : ((1.055 * powf(b, 0.4166666666666667)) - 0.055) * 255.;

  __syncthreads();
  pxl_rgb[idx] = round(r);
  pxl_rgb[idx + N] = round(g);
  pxl_rgb[idx + 2*N] = round(b);
}


#else
__global__ void convertRGB2Lab(
  float *pxl_rgb, float *pxl_lab_label, 
  int nCol, int nRow, int N)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  // aligned and coalesced reads
  float r = (float)pxl_rgb[idx] * 0.003921569F; // / 255.0F
  float g = (float)pxl_rgb[idx + N] * 0.003921569F;
  float b = (float)pxl_rgb[idx + 2*N] * 0.003921569F;

  float X = 0.412453f * r + 0.35758f  * g + 0.180423f * b; 
  float Y = 0.212671f * r + 0.71516f  * g + 0.072169f * b;
  float Z = 0.019334f * r + 0.119193f * g + 0.950227f * b;

  float L = cbrtf(Y);
  float fX = X * 1.052128F; // / nCIE_LAB_D65_xn; 
  float fY = L - 16.0F;
  float fZ = Z * 0.918482F; // / nCIE_LAB_D65_zn;

  L = 116.0F * L - 16.0F;
  float A = 500.0F * ((cbrtf(fX) - 16.0F) - fY);
  float B = 200.0F * (fY - (cbrtf(fZ) - 16.0F));  

  // Now scale Lab range
  L = L * 255.0F * 0.01F; // / 100.0F
  A = A + 128.0F;
  B = B + 128.0F;

  // force aligned and coalesced writes
  __syncthreads();
  pxl_lab_label[idx] = L;
  pxl_lab_label[idx + N] = A;
  pxl_lab_label[idx + 2*N] = B;

  // label with centroid index, -1 default value
  pxl_lab_label[idx + 3*N] = -1;

}


__global__ void convertLab2RGB(
  float *pxl_lab_label, float *pxl_rgb, 
  int nCol, int nRow, int N)
{
  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;
  
  // aligned and coalesced reads
  float L = pxl_lab_label[idx];
  float A = pxl_lab_label[idx + N];
  float B = pxl_lab_label[idx + 2*N];
  L = L * 100.0F * 0.003921569F;  // / 255.0F
  A = A - 128.0F;
  B = B - 128.0F;

  float P = (L + 16.0F) * 0.008621F; // / 116.0F
  float y = powf(P, 3.0F);
  float x = nCIE_LAB_D65_xn * powf((P + A * 0.002F), 3.0F); // / 500.0F
  float z = nCIE_LAB_D65_zn * powf((P - B * 0.005F), 3.0F); // / 200.0F

  float r = 3.240479F * x - 1.53715F  * y - 0.498535F * z; 
  r = r > 1.0f ? 1.0f : r;

  float g = -0.969256F * x + 1.875991F  * y + 0.041556F * z;
  g = g > 1.0f ? 1.0f : g;

  float b = 0.055648F * x - 0.204043F * y + 1.057311F * z;
  b = b > 1.0f ? 1.0f : b;


  r = (float)((unsigned char) (255.0F * r));
  g = (float)((unsigned char) (255.0F * g));
  b = (float)((unsigned char) (255.0F * b));
  
  // force aligned and coalesced writes
  __syncthreads();
  pxl_rgb[idx] = r;
  pxl_rgb[idx + N] = g;
  pxl_rgb[idx + 2*N] = b;
}

#endif

