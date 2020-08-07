
__global__ void convertRGB2Lab(float *pxl_rgb, float *pxl_lab_label, int nCol, int nRow, int N);

__global__ void convertLab2RGB(float *pxl_lab_label, float *pxl_rgb, int nCol, int nRow, int N);


