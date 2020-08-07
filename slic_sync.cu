#include "slic.h"
#include "converter.h"
#include "common/printutil.h"
#include "common/common.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>


/* To compile this program you need to launch nvcc as follows:
nvcc -D DEBUG -D SNAPSHOT -arch=sm_20 slic.cu cimage/imagebmp.c -o slic
nvcc -D DEBUG -D SNAPSHOT -O3 -arch=sm_20 slic.cu cimage/imagebmp.c -o slic
*/


/////////////////  UTILS variables, structs and imports

#ifdef DEBUG
int debug = 1;
#else
int debug = 0;
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

__constant__ int S;
__constant__ int N;
__constant__ float M;
__constant__ int K;
__constant__ int dx8[8] = {-1, -1, 0, 1, 1, 1, 0, -1};
__constant__ int dy8[8] = {0, -1, -1, -1, 0, 1, 1, 1};
 


/////////////////  UTILS Function conversion and print


/*  printClusters: Stampa per ogni cluster la posizione e il colore 

    esempio:   printClusters<<<(K_h + 32 - 1) / 32, 32>>>(d_cluster)
*/
__global__ void printClusters(float *cluster)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
 
  //bounds check
  if (ix >= K)
    return;

  float L = cluster[ix];
  float A = cluster[ix + K];
  float B = cluster[ix + 2*K];
  int x = (int)cluster[ix + 3*K];
  int y = (int)cluster[ix + 4*K];
  int cc = (int)cluster[ix + 5*K];
  printf("Centroid Id:%d - cc:%d x:%d y:%d - L:%2.6f A:%2.6f B:%2.6f\n", ix, cc, x, y, L, A, B);
}


/*  printLabel: Stampa la posizione di un pixel e la sua label, filtrando i pixel
                che si trovano in una certa posizione

    esempio:   printLabel<<<grid, block>>>(d_pxl_lab, nCol, nRow);
*/
__global__ void printLabel(float *pxl_label, int nCol, int nRow)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  int label = (int)pxl_label[idx + 3*N];
  if (ix % 255 == 0){
    printf("(pixel_X:%d pixel_Y:%d), label:%d\n", ix, iy, label);
  }
}


/*  convertCluster: Converti l'immagine e quindi i pixel, con il colore delle label
                    associate, oppure con 255. se non è presente nessuna label 

*/
__global__ void convertCluster(float *pxl_label, float *cluster, float *output_tmp, int nCol, int nRow)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  int label = (int)pxl_label[idx + 3*N];
  unsigned char p = (unsigned char)(label < 0);
  
  float L = p ? 255. : cluster[label];
  float A = p ? 255. : cluster[label + K];
  float B = p ? 255. : cluster[label + 2*K];

  __syncthreads();

  output_tmp[idx] = L;
  output_tmp[idx + N] = A;
  output_tmp[idx + 2*N] = B;
  output_tmp[idx + 3*N] = label;
}



/*  convertCSV: Converti l'immagine e quindi i pixel, con l'id delle label
                associate, e quindi salvalo in formato csv, con ogni riga che
                rappresenta una griglia dell'immagine.

*/

__global__ void convertCSV(float * pxl_label, int * csv, int nCol, int nRow){
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  int label = (int)pxl_label[idx + 3*N];
  csv[idx] = label;
}



/*  drawContours: questa funzione serve a disegnare i contorni
                  l'argomento è l'immagine già clusterizzata e i pixel originali
                  L'output è l'immagine originale con i contorni.
                  Lanciarla alla fine quando non si intende più utilizzare pxl_label

    esempio:  drawContours<<<grid, block>>>(d_output_tmp, nCol, nRow);
*/
__global__ void drawContours(float *pxl_rgb, float *output_tmp, int nCol, int nRow)
{
  int ix = threadIdx.x + blockIdx.x * blockDim.x;
  int iy = threadIdx.y + blockIdx.y * blockDim.y;
  int idx = iy*nCol + ix;

  // bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  // Calcola distanze
  int nr_pxl = 0;
  int l1, l2, i, j;

  // Compare dei pixel 8 neighbours
  for (int k = 0; k < 8; k++) {
    i = dx8[k] + ix;
    j = dy8[k] + iy;

    if (i >= 0 && i < nCol && j >= 0 && j < nRow) {
      l1 = output_tmp[(idx) + 3*N];
      l2 = output_tmp[(j * nCol + i) + 3*N];
      nr_pxl = l1 != l2 ? nr_pxl + 1 : nr_pxl;
    }
  }

  // Modifichiamo il colore del pixel se è un contorno
  if (nr_pxl >= 2) {
    pxl_rgb[idx] = 255. ;
    pxl_rgb[idx + N] = 255. ;
    pxl_rgb[idx + 2*N] = 255. ;
  } 

}



/////////////////  Main Cluster function

/*  initClusters: Inizializza ogni cluster, equidistanti SxS.
                  Sposta ogni cluster
*/
__global__ void initClusters(float *pxl_label, float *cluster, int width, int height, int sWidth)
{
  int centroid_Id = threadIdx.x + blockIdx.x * blockDim.x;

  // distribuisci l'id del centroide in quello della width frazionata rispetto ad S, vale soltanto all'inizio
  int ix = (((centroid_Id ) % sWidth) * S) + S;
  int iy = (((centroid_Id ) / sWidth) * S) + S;

  //bounds check
  if (centroid_Id >= K || ix >= width - S/2, iy >= height - S/2)
    return;
 
  // posizione pixel
  float min_gradient = INFINITY;
  int min_ix = ix, min_iy = iy;
  float L, L2, L3;
  unsigned char p;

  // sposta centroide su pixel che ha gradiente minore, su spazio 3x3
  for(int i = ix - 1; i < ix + 2; i++) {
    for(int j = iy - 1; j < iy + 2; j++) {
      L3 = pxl_label[j*width + i];
      L2 = pxl_label[j*width + i+1];
      L = pxl_label[(j+1)*width + i];
      
      p = (sqrtf(powf(L - L3, 2)) + sqrtf(powf(L2 - L3,2))) < min_gradient;
      min_gradient = p ? fabs(L - L3) + fabs(L2 - L3) : min_gradient;
      min_ix = p ? i : min_ix;
      min_iy = p ? j : min_iy;
    }
  }

  int idx = min_iy * width + min_ix;

  __syncthreads();

  L = pxl_label[idx];
  L2 = pxl_label[idx + N];
  L3 = pxl_label[idx + 2*N];

  cluster[centroid_Id] = L;
  cluster[centroid_Id + K] = L2;
  cluster[centroid_Id + 2*K] = L3;
  cluster[centroid_Id + 3*K] = (float)min_ix;
  cluster[centroid_Id + 4*K] = (float)min_iy;

}


/*  assignment: Lancia upperbound di N thread, ciascuno rappresentante un pixel
                E quindi confrontalo con i cluster che si trovano
                nel range 2S x 2S rispetto alla posizione del pixel
*/
__global__ void assignment(float *pxl_label, float *clusters, int nCol, int nRow)
{

  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  // imposta di default label -1 e distanza INF
  float L = pxl_label[idx];
  float A = pxl_label[idx + N];
  float B = pxl_label[idx + 2*N];
  float min_dist = INFINITY;
  int idx_centroid = -1;
  unsigned char p = 1;

  float L_c, A_c, B_c, X_c, Y_c;

  for (int i=0; i<K; i++) {

    __syncthreads();
    L_c = clusters[i];
    A_c = clusters[i + K];
    B_c = clusters[i + 2*K];
    X_c = clusters[i + 3*K];
    Y_c = clusters[i + 4*K];

    // calcola a prescindere D
    L_c = sqrtf(powf(L - L_c, 2) + powf(A - A_c, 2) + powf(B - B_c, 2));  // dc
    A_c = sqrtf(powf(ix - X_c, 2) + powf(iy - Y_c, 2));                   // ds
    B_c = sqrtf(powf(L_c / M, 2) + powf(A_c / (float)S, 2));              // D

    // Controlla che cluster si trovi nel range 2*S x 2*S
    p = (abs(X_c - ix) < S && abs(Y_c - iy) < S) && (min_dist > B_c);

    // Update label cluster, nel caso il valore di distanza 'D' è minore
    idx_centroid = p ? i : idx_centroid;
    min_dist = p ? B_c : min_dist;
  }
 
  pxl_label[idx + 3*N] = (float)idx_centroid;

}
 
 
/*  update: Ogni kernel parallelamente calcola la sua nuova posizione
            in base alla media dei punti, dei pixel che hanno la stessa label
            e che si trovano nel range 2S x 2S
*/
__global__ void update(float *pxl_label, float *clusters, int nCol, int nRow)
{
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  
  //bounds check
  if (idx >= K)
    return;

  float L_c=0, A_c=0, B_c=0;
  unsigned int x_c = (unsigned int)clusters[idx + 3*K];
  unsigned int y_c = (unsigned int)clusters[idx + 4*K];
  unsigned int ccounter = 0;
  unsigned int ccounter_x = 0;
  unsigned int ccounter_y = 0;

  // inizializza counters
  int idN, ix, iy;
  unsigned char p = 0;

  for (int i=x_c - S; i < x_c + S; i++) {
    ix = i < 0 ? 0 : i;

    for (int j=y_c - S; j < y_c + S; j++) {
      iy = j < 0 ? 0 : j;
      
      idN = iy * nCol + ix;

      // Aggiorna counters, se il pixel corrente ha la stessa label
      p = ((int)pxl_label[idN + 3*N]) == idx;

      if ( p ) {
        ccounter += 1;
        ccounter_x += ix;
        ccounter_y += iy;
        L_c += pxl_label[idN];
        A_c += pxl_label[idN + N];
        B_c += pxl_label[idN + 2*N];
      }

    }
  }

  if (ccounter == 0)
    return;

  // update nuova posizione cluster
  ix = (ccounter_x / ccounter);
  iy = (ccounter_y / ccounter);
  L_c /= ccounter;
  A_c /= ccounter;
  B_c /= ccounter;

  __syncthreads();
  clusters[idx] = L_c;
  clusters[idx + K] = A_c;
  clusters[idx + 2*K] = B_c;
  clusters[idx + 3*K] = (float)ix;
  clusters[idx + 4*K] = (float)iy;

}


/*  labelConnect: Questa funzione è un pezzo dell'assignment, serve per risolver
                  i pixel che non hanno ancora un cluster e quindi associarlo
                  non più con il range 2S x 2S, viene eseguita soltanto se c'è la flag dal main
*/
__global__ void labelConnect(float *pxl_label, float *clusters, int nCol, int nRow)
{

  unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
  unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
  unsigned int idx = iy*nCol + ix;

  //bounds check
  if (ix >= nCol || iy >= nRow)
    return;

  int label = (int)pxl_label[idx + 3*N];

  // esci se il pixel è già assegnato
  if ( label >= 0)
    return;

  // Questa parte è identica all'assignment, tranne che per l'if sulla posizione
  float L = pxl_label[idx];
  float A = pxl_label[idx + N];
  float B = pxl_label[idx + 2*N];
  float min_dist = INFINITY;
  int idx_centroid = -1;
  unsigned char p = 1;

  float L_c, A_c, B_c, X_c, Y_c;

  for (int i=0; i<K; i++) {

    __syncthreads();
    L_c = clusters[i];
    A_c = clusters[i + K];
    B_c = clusters[i + 2*K];
    X_c = clusters[i + 3*K];
    Y_c = clusters[i + 4*K];

    // calcola a prescindere D
    L_c = sqrtf(powf(L - L_c, 2) + powf(A - A_c, 2) + powf(B - B_c, 2));  // dc
    A_c = sqrtf(powf(ix - X_c, 2) + powf(iy - Y_c, 2));                   // ds
    B_c = sqrtf(powf(L_c / M, 2) + powf(A_c / (float)S, 2));              // D

    p = min_dist > B_c;
    idx_centroid = p ? i : idx_centroid;
    min_dist = p ? B_c : min_dist;
  }
 
  pxl_label[idx + 3*N] = (float)idx_centroid;

}
 



/////////////////  Function Main

void print_help(int argc, char **argv);
int writeCSV(int * h_csv, char * buff, int nCol, int nRow);


int main(int argc, char **argv) {

  if(argc < 4 ){
    print_help(argc, argv);
    return 1;
  }

  int cc = 0, nIteration = 10, snapshot = 0, contour = 0, label_connect = 0, csv_mode = 0;
  char *filename = argv[1];
  char out_filename_std[] = "out_lenna.bmp";
  char * out_filename = out_filename_std;

  for(int i=4; i<argc; i++) {
    if(!strcmp(argv[i], "-s")) {
      snapshot = 1;

    } else if(!strcmp(argv[i], "-c")) {
      contour = 1;

    } else if(!strcmp(argv[i], "-F")) {
      label_connect = 1;

    } else if(!strcmp(argv[i], "-T")) {
      csv_mode = label_connect = 1;

    } else if(!strcmp(argv[i], "-I")) {
      i++;
      nIteration = atoi(argv[i]);

    } else if(!strcmp(argv[i], "-o")) {
      i++;
      out_filename = argv[i];

    } else {
      fprintf(stderr, "Unknow parameter:%s ..Ignoring\n", argv[i]);
    }
  }

  if(nIteration < 0 || nIteration > 25){
    fprintf(stderr, "Number iteration cannot be negative or greater than 25 .. exit\n");
    return 1;
  }

  struct imgProp ip;

  char output_dir[] = "output_bmp"; //hardcoded in makefile for 'make clean'
  char buff[SIZE];

  pxl *output_rgb = readBMP(&ip, filename);
  int nRow = ip.hPixels, nCol = ip.wBytes / 3;
  int nElem = nRow * nCol;
  int nBytes = nElem * sizeof(float) * 3;

  if(debug) printY("Setting SLIC params, K,M, ..");
  int K_h = atoi(argv[2]);
  float M_h = atof(argv[3]);
  int S_h = sqrt(nElem / K_h);
  
  // check argument 
  unsigned char c = (K_h <= 0) << 2 | (M_h <= 0.) << 1 | (S_h <= 0);
  if ( c ) {
    if ( c & 1)
      fprintf(stderr, "S and so K");
    else if ( c & 2)
      fprintf(stderr, "M");
    else 
      fprintf(stderr, "K");

    fprintf(stderr, " cannot be zero or negative.. exit\n");
    return 1;
  }

  if ( K_h <= 0){
      fprintf(stderr, "Number iteration cannot be negative or greater than 20 .. exit\n");
      return 1;
  }

  CHECK(cudaMemcpyToSymbol(N, &nElem, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(S, &S_h, sizeof(int)));
  CHECK(cudaMemcpyToSymbol(M, &M_h, sizeof(float)));
  CHECK(cudaMemcpyToSymbol(K, &K_h, sizeof(int)));

  /* Converti l'array da unsigned char a float, 
     e intanto visto la sequenzialità, converti l'array in SoA
  */
  float * output = (float *)malloc(nBytes);
  for (int i=0; i<nElem; i++) {
    *(output + i) = (float)((unsigned int)output_rgb[i*3]);
    *(output + nElem + i) = (float)((unsigned int)output_rgb[i*3 + 1]);
    *(output + 2*nElem + i) = (float)((unsigned int)output_rgb[i*3 + 2]);
  }

  if(debug) printY("Allocating device memory");
  float *d_pxl_rgb, *d_pxl_lab, *d_output_tmp, *d_cluster;
  CHECK(cudaMalloc((float **)&d_pxl_rgb, nBytes ));
  CHECK(cudaMalloc((float **)&d_output_tmp, 4 * nBytes / 3 ));
  CHECK(cudaMalloc((float **)&d_pxl_lab, 4 * nBytes / 3));
  CHECK(cudaMalloc((float **)&d_cluster, 5 * K_h * sizeof(float)));

  CHECK(cudaMemcpy(d_pxl_rgb, output, nBytes, cudaMemcpyHostToDevice));

  if(debug) printY("Preparing the global variables before launching kernels");
  int dimx = 16, dimy = 16;
  dim3 block (dimx, dimy);
  dim3 grid ((nCol + block.x - 1) / block.x, (nRow + block.y - 1) / block.y);

  if(debug) printY("Launching convertRGB2Lab"); 
  convertRGB2Lab<<<grid, block>>>(d_pxl_rgb, d_pxl_lab, nCol, nRow, nElem);
  if(debug) printG("Done"); 
  
  printf("\n#####  Summary: #######\n"); 
  printf("  image params -> width(bitmap aligned):%d height:%d nElem:%d\n", nCol, nRow, nElem);
  printf("  cluster params -> K:%d M:%2.2f S:%d Iteration:%d\n", K_h, M_h, S_h, nIteration);
  printf("  Debugmode:%d snapshotmode:%d\n", debug, snapshot);
  printf("  The snapshots are saved into folder %s with format '<current_iteration>_<M>_<K>_<filename>'\n", output_dir);
  printf("  The last snapshot is always saved as %s in current workdir\n\n", out_filename);

  if(debug){
    printY("Launching the kernels"); 
    printf("  initClusters<<< grid(%d,%d) block(%d,%d) >>>\n", (K_h+dimx -1)/dimx, 1, dimx, 1);
  }

  initClusters<<<(K_h + dimx - 1) / dimx, dimx>>>(d_pxl_lab, d_cluster, nCol, nRow, nCol / S_h);
  //printClusters<<<(K_h + dimx - 1) / dimx, dimx>>>(d_cluster);

  if(debug) printY("Clusters initialized. Start iterating the assignment and update of the clusters"); 


  for (cc = 1; cc <= nIteration; cc++) {
    CHECK(cudaDeviceSynchronize());
    if (debug) printf("  assignment<<< grid(%d,%d) block(%d,%d) >>>\n", grid.x, grid.y, block.x, block.y);
    assignment<<<grid, block>>>(d_pxl_lab, d_cluster, nCol, nRow);

    if (debug) printf("  update<<< grid(%d,%d) block(%d,%d) >>>\n", (K_h+dimx-1)/dimx, 0,dimx, 0);
    update<<<(K_h + dimx - 1) / dimx, dimx>>>(d_pxl_lab, d_cluster, nCol, nRow);

    if(snapshot && cc < nIteration) {
      memset(buff, 0, SIZE);
      snprintf(buff, SIZE, "%s/iter%d_M%d_K%d_%s", output_dir, cc, (int)M_h, K_h, filename);

      convertCluster<<<grid, block>>>(d_pxl_lab, d_cluster, d_output_tmp, nCol, nRow);
      convertLab2RGB<<<grid, block>>>(d_output_tmp, d_pxl_rgb, nCol, nRow, nElem);
      CHECK(cudaMemcpy(output, d_pxl_rgb, nBytes, cudaMemcpyDeviceToHost));

      for (int i=0; i<nElem; i++) {
        output_rgb[i*3] = (pxl)*(output + i);
        output_rgb[i*3 + 1] = (pxl)*(output + nElem + i);
        output_rgb[i*3 + 2] = (pxl)*(output + 2*nElem + i);
      }
      writeBMP(&ip, output_rgb, buff);
    }
  }

  if(debug) printG("Done");

  if(label_connect) labelConnect<<<grid, block>>>(d_pxl_lab, d_cluster, nCol, nRow);


  if(!csv_mode) {
    convertCluster<<<grid, block>>>(d_pxl_lab, d_cluster, d_output_tmp, nCol, nRow);

    if (contour) {
      convertLab2RGB<<<grid, block>>>(d_pxl_lab, d_pxl_rgb, nCol, nRow, nElem);
      drawContours<<<grid, block>>>(d_pxl_rgb, d_output_tmp, nCol, nRow);
    } else {
      convertLab2RGB<<<grid, block>>>(d_output_tmp, d_pxl_rgb, nCol, nRow, nElem);
    }

    CHECK(cudaMemcpy(output, d_pxl_rgb, nBytes, cudaMemcpyDeviceToHost));

    for (int i=0; i<nElem; i++) {
      output_rgb[i*3] = (pxl)((unsigned int)*(output + i));
      output_rgb[i*3 + 1] = (pxl)((unsigned int)*(output + nElem + i));
      output_rgb[i*3 + 2] = (pxl)((unsigned int)*(output + 2*nElem + i));
    }

    writeBMP(&ip, output_rgb, out_filename);

  } else {

    if(debug) printY("CSV mode");

    int *d_csv;
    int *h_csv = (int *) malloc(sizeof(int) * nBytes/3);

    CHECK(cudaMalloc((int **)&d_csv, nBytes/3 ));
    convertCSV<<<grid, block>>>(d_pxl_lab, d_csv, nCol, nRow);
    CHECK(cudaMemcpy(h_csv, d_csv, nBytes/3, cudaMemcpyDeviceToHost));
    
    writeCSV(h_csv, out_filename, nCol, nRow);

    CHECK(cudaFree(d_csv));
    free(h_csv);
  }


  // Libera memoria device
  if(debug) printG("Freeing device memory");
  CHECK(cudaFree(d_pxl_rgb));
  CHECK(cudaFree(d_pxl_lab));
  CHECK(cudaFree(d_cluster));
  CHECK(cudaFree(d_output_tmp));

  // Libera memoria host
  if(debug) printG("Freeing host memory");
  free(output);
  free(output_rgb);

  // reset device
  cudaDeviceReset();

  if(debug) printG("Done, quitting...");
  return 0;
}


// schermata comandi opzioni
void print_help(int argc, char **argv) {
  fprintf(stderr, "\nUSAGE: %s <parameters> [optional parameters]\n\n", argv[0]);
  fprintf(stderr, "\tI parametri obbligatori <parameters> devono seguire per forza il seguente ordine:\n");
  fprintf(stderr, "\t<filename.bmp> <k-centroid_number> <M_scale>\n\n");
  fprintf(stderr, "\tI parametri opzionali [ ... ] possono essere i seguenti:\n");
  fprintf(stderr, "\t  -I <niteration>: Se non viene settato il numero di iterazioni di default è 10, come nel paper\n");
  fprintf(stderr, "\t  -s             : Flag che serve per salvare ogni snaphost prodotto in ogni iterazione\n\n");
  fprintf(stderr, "\t  -o <output>    : Se non viene settato l'output di default viene salvato come 'out_lenna.bmp'\n\n");
  fprintf(stderr, "\t  -c             : Flag che serve per salvare l'output come l'immagine originale ma con i contorni\n\n");
  fprintf(stderr, "\t  -F             : Forza label connectivity, per i pixel che non hanno ancora associato un cluster alla fine delle iterazioni\n\n");
  fprintf(stderr, "\t  -T             : Implica anche '-F', e serve per salvare l'output come CSV, ogni riga rappresente una row dell'immagine con valori però le label, NON I PIXEL!\n");
  fprintf(stderr, "\t                   serve per superpixel-benchmark (D. Stutz, url:www.github.com/davidstutz/superpixel-benchmark)\n");
  fprintf(stderr, "\t                   Inserire anche '-o <output.csv>' \n\n");
  fprintf(stderr, "ESEMPIO: %s lenna.bmp 400 40 -s -I 10\n\n", argv[0]);
}


// salva in formato csv, leggi flag '-T' su help
int writeCSV(int * h_csv, char * buff, int nCol, int nRow) {

  FILE * fp = fopen(buff, "wa");

  if ( !fp ) {
    fprintf(stderr, "Cannot open CSV file to append .. returing main\n");
    return 0;
  }

  int size = 11 * nCol * sizeof(char);
  char * row_buff = (char *)malloc(size);
  int i, j, cc;

  for(i=0, cc=0; i<nRow-1; i++, cc=0) {
    memset(row_buff, 0, size);
    cc += snprintf(row_buff, size, "%d", h_csv[i*nCol]);

    for(j=1; j<nCol; j++) {
      cc += snprintf((row_buff+cc), size, ",%d", h_csv[i*nCol+j]);
    }

    fprintf(fp, "%s\n", row_buff);
  }

  memset(row_buff, 0, size);
  cc += snprintf(row_buff, size, "%d", h_csv[i*nCol]);
  for(j=1; j<nCol; j++) {
      cc += snprintf((row_buff+cc), size, ",%d", h_csv[i*nCol+j]);
  }
  fprintf(fp, "%s", row_buff);

  free(row_buff);
  fclose(fp);

  return 1;
}

