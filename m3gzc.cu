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
    //else printf("success\n");
}

int *m3gzcGPU(SerializedSampleSet sss1,SerializedSampleSet sss2){
    clock_t timer;
    clock_t timer0;
    float time;
    TIMER_BEGIN(timer);
    TIMER_BEGIN(timer0);
    cudaEvent_t ev0,ev1;

    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    
    //sss1.print();
    //sss2.print();

    cudaSetDevice(0);

    cudaEventRecord(ev0,0);

    cudaError_t cudaerr;
    int attr=0;

    cudaerr=cudaDeviceGetAttribute(&attr,cudaDevAttrMaxSharedMemoryPerBlock,0);
    if (cudaerr != CUDA_SUCCESS) 
        printf("error \"%s\".\n", cudaGetErrorString(cudaerr));
    //cout<<"Max shared mem per block(bytes): "<<attr<<endl;

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
    TIMER_PRINT("pre",timer);


    TIMER_BEGIN(timer);
    cudaMalloc(&d_sss1,sssSize);
    cudaMalloc(&d_sss2,sssSize);
    cudaMalloc(&d_resultMat,resultSize*sizeof(float));
    cudaMalloc((void**)&d_test_data,test_data_length*sizeof(Data_Node));
    cudaMalloc(&d_test_data_length,sizeof(int));
    TIMER_PRINT("malloc",timer);

    TIMER_BEGIN(timer);
    cudaMemcpy(d_sss1,&sss1,sssSize,cudaMemcpyHostToDevice);
    cudaMemcpy(d_sss2,&sss2,sssSize,cudaMemcpyHostToDevice);
    cudaMemcpy((void*)d_test_data,test_data,test_data_length*sizeof(Data_Node),cudaMemcpyHostToDevice);
    cudaMemcpy(d_test_data_length,&test_data_length,sizeof(int),cudaMemcpyHostToDevice);
    TIMER_PRINT("mcopy",timer);

    TIMER_BEGIN(timer);
    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    dim3 dimGrid((sss1.numSample-1)/BLOCK_SIZE+1,(sss2.numSample-1)/BLOCK_SIZE+1);

    TIMER_PRINT("pre gzc compute",timer);

    TIMER_BEGIN(timer);
    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("before kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        //else printf("before kernel launch success\n");
    }
    TIMER_PRINT("error check",timer);

    TIMER_BEGIN(timer);
    m3gzcKernel<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sss1,d_sss2,d_resultMat);
    //m3gzcKernelWithSharedMemory<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sss1,d_sss2,d_resultMat);
    TIMER_PRINT("gzc compute",timer);
    TIMER_BEGIN(timer);
    reportError();

    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        //else printf("kernel launch success\n");
    }
    TIMER_PRINT("error check",timer);

    TIMER_BEGIN(timer);
    cudaMemcpy(resultMat,d_resultMat,resultSize*sizeof(float),cudaMemcpyDeviceToHost);

    TIMER_PRINT("mcopy",timer);
    //for(int i=0;i<resultSize;i++){
    //    if((i)%sss2.numSample==0) cout<<endl;
    //    cout<<resultMat[i]<<'\t';
    //}
    //cout<<endl;

    //for(int i=0;i<resultSize;i++){
    //    if((i)%sss2.numSample==0 && i>0) cout<<endl;
    //    cout<<resultMat[i]<<'\t';
    //}

    TIMER_BEGIN(timer);
    int *resultArray= new int[sss1.numSample];
    int *d_resultArray;
    cudaMalloc(&d_resultArray,sss1.numSample*sizeof(int));

    int threadsPerBlock=128;
    int blockPerGrid=(sss1.numSample-1)/threadsPerBlock+1;
    TIMER_PRINT("malloc",timer);

    TIMER_BEGIN(timer);
    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("before kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        //else printf("before kernel launch success\n");
    }
    TIMER_PRINT("error check",timer);
    TIMER_BEGIN(timer);
    minmaxKernel<<<blockPerGrid,threadsPerBlock>>>(d_resultMat,sss1.numSample,sss2.numSample,d_resultArray);
    TIMER_PRINT("min kernel",timer);

    TIMER_BEGIN(timer);
    reportError();

    {
        cudaError_t cudaerr = cudaDeviceSynchronize();
        if (cudaerr != CUDA_SUCCESS) 
            printf("kernel launch failed with error \"%s\".\n", cudaGetErrorString(cudaerr));
        //else printf("kernel launch success\n");
    }
    TIMER_PRINT("error check",timer);

    TIMER_BEGIN(timer);
    cudaMemcpy(resultArray,d_resultArray,sss1.numSample*sizeof(int),cudaMemcpyDeviceToHost);
    TIMER_PRINT("post copy",timer);

    //cout<<sss1.numSample<<' '<<sss2.numSample<<endl;
    //for(int i=0;i<sss1.numSample;i++){
    //    cout<<i<<' '<<resultArray[i]<<endl;
    //}
    //cout<<endl;

    TIMER_BEGIN(timer);

    cudaFree(d_sss1);
    cudaFree(d_sss2);
    cudaFree(d_resultMat);
    cudaFree(d_test_data);
    cudaFree(d_test_data_length);
    cudaFree(d_resultArray);

    delete test_data;

    cudaEventRecord(ev1,0);
    cudaEventSynchronize(ev0);
    cudaEventSynchronize(ev1);

    cudaEventElapsedTime(&time,ev0,ev1);
    cout<<"cuda event time "<<time<<endl;

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);

    TIMER_PRINT("post compute",timer);
    TIMER_PRINT("total compute time",timer0);
    return resultArray;
}

float getDistance2(Data_Sample test_sample,const SerializedSampleSet sss,int i){
    int dataIndexBegin,dataIndexEnd;
    dataIndexBegin=i==0?0:sss.dataNodeOffsetArray[i-1];
    dataIndexEnd=sss.dataNodeOffsetArray[i];
    int j=0,k=dataIndexBegin;
    float sum=0;
    //cout<<endl<<dataIndexBegin<<endl<<dataIndexEnd<<endl;
    while(1){
        //cout<<j<<endl<<k<<endl;
        if(j==test_sample.data_vector_length && k==dataIndexEnd) break;
        if(k==dataIndexEnd) {
            sum+=SQUARE(test_sample.data_vector[j].value);
            j++;
        } else if(j==test_sample.data_vector_length) {
            sum+=SQUARE(sss.dataNodeValueArray[k]);
            k++;
        } else if(test_sample.data_vector[j].index < sss.dataNodeIndexArray[k]){
            sum+=SQUARE(test_sample.data_vector[j].value);
            j++;
        } else if(test_sample.data_vector[j].index > sss.dataNodeIndexArray[k]){
            sum+=SQUARE(sss.dataNodeValueArray[k]);
            k++;
        } else {
            float tmp=test_sample.data_vector[j].value-sss.dataNodeValueArray[k];
            sum+=SQUARE(tmp);
            //cout<<sum<<endl;
            j++;
            k++;
        }
    }
    return sum;
}

float getDistance2(const SerializedSampleSet sss1,int i,const SerializedSampleSet sss2,int j){
    int dataIndexBegin1,dataIndexEnd1;
    dataIndexBegin1=i==0?0:sss1.dataNodeOffsetArray[i-1];
    dataIndexEnd1=sss1.dataNodeOffsetArray[i];
    int dataIndexBegin2,dataIndexEnd2;
    dataIndexBegin2=j==0?0:sss2.dataNodeOffsetArray[j-1];
    dataIndexEnd2=sss2.dataNodeOffsetArray[j];
    int ii=dataIndexBegin1,ij=dataIndexBegin2;
    float sum=0;
    while (1){
        if(ii==dataIndexEnd1 && ij==dataIndexEnd2) break;
        else if(ii==dataIndexEnd1) {
            sum+=SQUARE(sss2.dataNodeValueArray[ij]);
            ij++;
        }else if (ij==dataIndexEnd2) {
            sum+=SQUARE(sss1.dataNodeValueArray[ii]);
            ii++;
        }else if (sss1.dataNodeIndexArray[ii]<sss2.dataNodeIndexArray[ij]){
            sum+=SQUARE(sss1.dataNodeValueArray[ii]);
            ii++;
        }else if (sss1.dataNodeIndexArray[ii]>sss2.dataNodeIndexArray[ij]){
            sum+=SQUARE(sss2.dataNodeValueArray[ij]);
            ij++;
        } else {
            float tmp=sss2.dataNodeValueArray[ij]-sss1.dataNodeValueArray[ii];
            sum+=SQUARE(tmp);
            ij++;
            ii++;
        } 
    }
    return sum;
}


int *m3gzcCPU(SerializedSampleSet sss1,SerializedSampleSet sss2){
    //sss1.print();
    //sss2.print();

    float * sumArray1,*sumArray2;
    int *resultArray;
    sumArray1=new float[sss1.numSample];
    sumArray2=new float[sss2.numSample];
    resultArray=new int[sss1.numSample];
    
    Data_Node * test_data=new Data_Node;
    test_data->index=1;
    test_data->value=1;

    Data_Sample test_sample;
    test_sample.index=0;
    test_sample.label=0;
    test_sample.data_vector_length=1;
    test_sample.data_vector=test_data;

    for(int i=0;i<sss1.numSample;i++) sumArray1[i]=getDistance2(test_sample,sss1,i);
    cout<<sumArray1[0]<<endl;
    for(int i=0;i<sss2.numSample;i++) sumArray2[i]=getDistance2(test_sample,sss2,i);

    for(int i=0;i<sss1.numSample;i++){
        float min=1;
        for(int j=0;j<sss2.numSample;j++) {
            float sum0=getDistance2(sss1,i,sss2,j);
            //cout<<i<<' '<<j<<' '<<sum0<<endl;
            float v=exp(-4*sumArray1[i]/sum0)-exp(-4*sumArray2[j]/sum0);
            //cout<<i<<' '<<j<<' '<<v<<endl;
            if (v<min) min=v;
        }

        //cout<<i<<' '<<min<<endl;

        if(min>THRESHOLD) resultArray[i]=1;
        else if(min<-THRESHOLD) resultArray[i]=-1;
        else resultArray[i]=0;
    }

    delete test_sample.data_vector;
    return resultArray;
}
