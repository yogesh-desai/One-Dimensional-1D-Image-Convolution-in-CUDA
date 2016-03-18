
#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>
#include<cuda_runtime_api.h>
#define funcCheck(stmt) do {                                                  \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            printf( "Failed to run stmt %d ", __LINE__);                      \
            printf( "Got CUDA error ...  %s ", cudaGetErrorString(err));      \
            return -1;                                                        \
        }                                                                     \
    } while(0)


__global__ void convolution_1D(float *N,float *M,float *P,int Mask_width,int width)
{
int i=blockIdx.x*blockDim.x+threadIdx.x;
float Pvalue=0.0;
int N_start_point=i-(Mask_width/2);
for(int j=0;j<Mask_width;j++)
{
 if(((N_start_point+j)>=0) && ((N_start_point+j)<width))
 {
  Pvalue+=N[N_start_point+j]*M[j];
 }
}
P[i]=Pvalue;
}


int main()
{
 float * input;
 float * Mask;
 float * output;
 float * device_input;
 float * device_Mask;
 float * device_output;
 int Mask_width=3;
 int width=5;

 input=(float *)malloc(sizeof(float)*width);
 Mask=(float *)malloc(sizeof(float)*Mask_width);
 output=(float *)malloc(sizeof(float)*width);
 for(int i=0;i<width;i++)
 {
  input[i]=1.0;
 }
 for(int i=0;i<Mask_width;i++)
 {
  Mask[i]=1.0;
 }

 printf("\nInput:\n");
 for(int i=0;i<width;i++)
 {
   printf("%0.2f ",input[i]);
 }printf("\n");

 printf("\nMask:\n");

  for(int i=0;i<Mask_width;i++)
  {
    printf("%0.2f ",Mask[i]);
  }printf("\n");

  funcCheck(cudaMalloc((void **)&device_input,sizeof(float)*width));
  funcCheck(cudaMalloc((void **)&device_Mask,sizeof(float)*Mask_width));
  funcCheck(cudaMalloc((void **)&device_output,sizeof(float)*width));

  funcCheck(cudaMemcpy(device_input,input,sizeof(float)*width,cudaMemcpyHostToDevice));
  funcCheck(cudaMemcpy(device_Mask,Mask,sizeof(float)*Mask_width,cudaMemcpyHostToDevice));

  dim3 dimGrid(((width-1)/Mask_width)+1, 1,1);
  dim3 dimBlock(Mask_width,1, 1);

  convolution_1D<<<dimGrid,dimBlock>>>(device_input,device_Mask,device_output,Mask_width,width);

  cudaError_t err1 = cudaPeekAtLastError();
  cudaDeviceSynchronize();

  printf( "Got CUDA error ... %s \n", cudaGetErrorString(err1));
  funcCheck(cudaMemcpy(output,device_output,sizeof(float)*width,cudaMemcpyDeviceToHost));


 printf("\n\nOutput: \n");

 for(int i=0;i<width;i++)
  {
   printf(" %0.2f \t",*(output+i));
  }

 cudaFree(device_input);
 cudaFree(device_Mask);
 cudaFree(device_output);

 free(input);
 free(output);
 free(Mask);


 printf("\n \nNumber of Blocks Created :%d",(((width-1)/Mask_width)+1));
 printf("\n \nNumber of Threads Per Block created in code: %d",(Mask_width));

 return 0;
}

 
