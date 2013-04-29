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

int *m3gzcGPU(SerializedSampleSet sss1,SerializedSampleSet sss2){

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

    //for(int i=0;i<resultSize;i++){
    //    if((i)%sss2.numSample==0 && i>0) cout<<endl;
    //    cout<<resultMat[i]<<'\t';
    //}

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

    delete test_data;

    return resultArray;
}

float getDistance2(Data_Sample test_sample,const SerializedSampleSet sss,int i){
    int dataIndexBegin,dataIndexEnd;
    dataIndexBegin=i==0?0:sss.dataNodeIndexArray[i-1];
    dataIndexEnd=dataNodeIndexArray[i];
    int j=0,k=dataIndexBegin;
    float sum=0;
    while(1){
        if(j==test_sample.data_vector_length && k==dataIndexEnd) break;
        if(k==dataIndexEnd) {
            sum+=SQUARE(test_sample.data_vector[j].value);
            j++;
        } else if(j==test_sample.data_vector_length) {
            sum+=SQUARE(sss.dataNodeValueArray[k]);
            k++;
        } if(test_sample.data_vector[j].index < sss.dataNodeIndexArray[k]){
            sum+=SQUARE(test_sample.data_vector[j].value);
            j++;
        } else if(test_sample.data_vector[j].index < sss.dataNodeIndexArray[k]){
            sum+=SQUARE(sss.dataNodeValueArray[k]);
            k++;
        } else {
            sum+=SQUARE(test_sample.data_vector[j].value)+SQUARE(sss.dataNodeValueArray[k]);
            j++;
            k++;
        }
    }
    return sum;
}


int *m3gzcCPU(SerializedSampleSet sss1,SerializedSampleSet sss2){
    float * sumArray1,*sumArray2,*resultArray;
    sumArray1=new float[sss1.numSample];
    sumArray2=new float[sss2.numSample];
    resultArray=new float[sss1.numSample];
    
    Data_Node * test_data=new Data_Node,*d_test_data;
    test_data->index=1;
    test_data->value=1;

    Data_Sample test_sample;
    test_sample.index=0;
    test_sample.label=0;
    test_sample.data_vector_length=1;
    test_sample.data_vector=test_data;

    for(int i=0;i<sss1.numSample;i++) sumArray1[i]=getDistance2(test_sample,sss1,i);
    for(int i=0;i<sss2.numSample;i++) sumArray2[i]=getDistance2(test_sample,sss2,i);

    for(int i=0;i<sss1.numSample;i++){
        float min=1;
        for(int j=0;j<sss2.numSample;j++) {
            float v=
            
    }

    delete test_sample.data_vector;
    delete test_sample;
}

