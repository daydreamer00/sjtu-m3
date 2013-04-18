#include"m3gzc_kernel.cu"
#include"m3gzc.h"
#include<cuda.h>
#include<stdio.h>

int m3gzc(SerializedSampleSet sss1,SerializedSampleSet sss2){

    cudaPrintfInit();
    cudaPrintfDisplay(stdout,true);

    Data_Node * test_data=new Data_Node,*d_test_data;
    test_data->index=1;
    test_data->value=3;

    int test_data_length=1;
    int * d_test_data_length;

    int resultSize=SerializedSampleSet::max_num_sample*SerializedSampleSet::max_num_sample;
    int *resultMat=new int[resultSize];
    int * d_resultMat;
    int BLOCK_SIZE=16;

    SerializedSampleSet *d_sss1,*d_sss2;

    size_t sssSize=sizeof(SerializedSampleSet);

    cudaMalloc(&d_sss1,sssSize);
    cudaMalloc(&d_sss2,sssSize);
    cudaMalloc(&d_resultMat,resultSize*sizeof(int));
    cudaMalloc((void**)&d_test_data,test_data_length*sizeof(Data_Node));
    cudaMalloc(&d_test_data_length,sizeof(int));

    cudaMemcpy(d_sss1,&sss1,sssSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sss2,&sss2,sssSize,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_test_data,test_data,test_data_length*sizeof(Data_Node),cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_data_length,&test_data_length,sizeof(int),cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid(sss1.numSample/BLOCK_SIZE,sss2.numSample/BLOCK_SIZE);
    m3gzcKernel<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sss1,d_sss2,d_resultMat);

    cudaMemcpy(resultMat,d_resultMat,resultSize*sizeof(int),cudaMemcpyDeviceToHost);

    cudaPrintfEnd();
}


