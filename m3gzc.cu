#include"m3gzc_kernel.cu"
#include"m3gzc.h"
#include<cuda.h>
#include<stdio.h>
#include<iostream>

using namespace std;

bool InitGPUSet()
{
    char GPU[100] = "GPU: ";
    cudaDeviceProp tCard;
    int num = 0;
    if(cudaSuccess == cudaGetDeviceCount(&num))
    {
        for(int i = 0; i < num; ++ i)
        {
            cudaSetDevice(i);
            cudaGetDeviceProperties(&tCard, i);
            puts(tCard.name);
        }
    }
    else  return false;
    return true;
}

bool cuPrintInit()
{
    cudaError_t err = cudaPrintfInit();
    if(0 != strcmp("no error", cudaGetErrorString(err)))  return false;
    return true;
}

void reportError(){
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != CUDA_SUCCESS) 
        printf("error \"%s\".\n", cudaGetErrorString(cudaerr));
    else printf("success\n");
}

int *m3gzc(SerializedSampleSet sss1,SerializedSampleSet sss2){

    sss1.print();
    sss2.print();

    cudaSetDevice(0);

    cudaError_t cudaerr;
    int attr=0;

    cudaerr=cudaDeviceGetAttribute(&attr,cudaDevAttrMaxSharedMemoryPerBlock,0);
    if (cudaerr != CUDA_SUCCESS) 
        printf("error \"%s\".\n", cudaGetErrorString(cudaerr));
    cout<<"Max shared mem per block(bytes): "<<attr<<endl;

    Data_Node * test_data=new Data_Node,*d_test_data;
    test_data->index=1;
    test_data->value=1;

    int test_data_length=1;
    int * d_test_data_length;

    int resultSize=sss1.numSample*sss2.numSample;//SerializedSampleSet::max_num_sample*SerializedSampleSet::max_num_sample;
    float *resultMat=new float[resultSize];
    float * d_resultMat;
    //int BLOCK_SIZE=16;

    SerializedSampleSet *d_sss1,*d_sss2;

    size_t sssSize=sizeof(SerializedSampleSet);

    cudaMalloc(&d_sss1,sssSize);
    cudaMalloc(&d_sss2,sssSize);
    cudaMalloc(&d_resultMat,resultSize*sizeof(float));
    cudaMalloc((void**)&d_test_data,test_data_length*sizeof(Data_Node));
    cudaMalloc(&d_test_data_length,sizeof(int));

    cudaMemcpy(d_sss1,&sss1,sssSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sss2,&sss2,sssSize,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_test_data,test_data,test_data_length*sizeof(Data_Node),cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_data_length,&test_data_length,sizeof(int),cudaMemcpyHostToDevice);

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid((sss1.numSample-1)/BLOCK_SIZE+1,(sss2.numSample-1)/BLOCK_SIZE+1);

    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("before kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        else printf("before kernel launch success\n");
    }

    //m3gzcKernel<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sss1,d_sss2,d_resultMat);
    m3gzcKernelWithSharedMemory<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sss1,d_sss2,d_resultMat);

    reportError();

    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        else printf("kernel launch success\n");
    }

    cudaMemcpy(resultMat,d_resultMat,resultSize*sizeof(float),cudaMemcpyDeviceToHost);

    //for(int i=0;i<resultSize;i++){
    //    if((i)%sss2.numSample==0) cout<<endl;
    //    cout<<resultMat[i]<<'\t';
    //}
    //cout<<endl;

    for(int i=0;i<16;i++){
        if((i)%sss2.numSample==0 && i>0) cout<<endl;
        cout<<resultMat[i]<<'\t';
    }
    cout<<endl;

    int *resultArray= new int[sss1.numSample];
    int *d_resultArray;
    cudaMalloc(&d_resultArray,sss1.numSample*sizeof(int));

    int threadsPerBlock=128;
    int blockPerGrid=(sss1.numSample-1)/threadsPerBlock+1;

    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("before kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        else printf("before kernel launch success\n");
    }
    minmaxKernel<<<blockPerGrid,threadsPerBlock>>>(d_resultMat,sss1.numSample,sss2.numSample,d_resultArray);

    reportError();

    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        else printf("kernel launch success\n");
    }

    cudaMemcpy(resultArray,d_resultArray,sss1.numSample*sizeof(int),cudaMemcpyDeviceToHost);

    for(int i=0;i<sss1.numSample;i++){
        cout<<resultArray[i]<<' ';
    }
    cout<<endl;

    cudaFree(d_sss1);
    cudaFree(d_sss2);
    cudaFree(d_resultMat);
    cudaFree(d_test_data);
    cudaFree(d_test_data_length);
    cudaFree(d_resultArray);

    return resultArray;
}


