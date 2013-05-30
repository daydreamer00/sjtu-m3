#include"m3gzc_kernel.cu"
#include"m3gzc.h"
#include<cuda.h>
#include<stdio.h>
#include<iostream>
#include<fstream>
#include<string>

using namespace std;

void reportError(){
    cudaError_t cudaerr = cudaGetLastError();
    if (cudaerr != CUDA_SUCCESS) {
        printf("error \"%s\".\n", cudaGetErrorString(cudaerr));
        exit(1);
    }
    //else printf("success\n");
}

void recordTime(cudaEvent_t &evtBegin,cudaEvent_t &evtEnd,string message){
    cudaEventRecord(evtEnd,0);
    float time;
    cudaEventSynchronize(evtBegin);
    cudaEventSynchronize(evtEnd);
    cudaEventElapsedTime(&time,evtBegin,evtEnd);
    fbreakdown<<message<<'\t'<<time<<endl;
    cudaEventRecord(evtBegin,0);
}

int *m3gzcGPU(SerializedSampleSet sss1,SerializedSampleSet sss2,SerializedSampleSet sss3){
    clock_t timer;
    clock_t timer0;
    float time;
    TIMER_BEGIN(timer);
    TIMER_BEGIN(timer0);
    cudaEvent_t ev0,ev1,ev2,ev3;

    cudaEventCreate(&ev0);
    cudaEventCreate(&ev1);
    cudaEventCreate(&ev2);
    cudaEventCreate(&ev3);
    
    //sss1.print();
    //sss2.print();
    //sss3.print();

    cudaSetDevice(0);

    cudaEventRecord(ev0,0);
    cudaEventRecord(ev2,0);

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

    int resultSize=sss1.numSample*sss2.numSample*sss3.numSample;//SerializedSampleSet::max_num_sample*SerializedSampleSet::max_num_sample;
    float *resultMat=new float[resultSize];
    float * d_resultMat;
    //int BLOCK_SIZE=16;

    SerializedSampleSet d_sss1,d_sss2,d_sss3;
    SerializedSampleSet *h_ssspArray[3],d_sssArray[3];
    h_ssspArray[0]=&sss1;
    h_ssspArray[1]=&sss2;
    h_ssspArray[2]=&sss3;

    size_t sssSize=sizeof(SerializedSampleSet);
    TIMER_PRINT("pre",timer);
    recordTime(ev2,ev3,"preprocessing 1");

    TIMER_BEGIN(timer);
    
    for(int i=0;i<3;i++){
        d_sssArray[i].numSample=h_ssspArray[i]->numSample;
        cudaMalloc(&(d_sssArray[i].labelArray),sizeof(float)*MAX_NUM_SAMPLE);
        cudaMalloc(&(d_sssArray[i].dataNodeOffsetArray),sizeof(int)*MAX_NUM_SAMPLE);
        cudaMalloc(&(d_sssArray[i].dataNodeIndexArray),sizeof(int)*MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE);
        cudaMalloc(&(d_sssArray[i].dataNodeValueArray),sizeof(float)*MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE);
        cudaMemcpy(d_sssArray[i].labelArray,h_ssspArray[i]->labelArray,sizeof(float)*MAX_NUM_SAMPLE,cudaMemcpyHostToDevice);
        cudaMemcpy(d_sssArray[i].dataNodeOffsetArray,h_ssspArray[i]->dataNodeOffsetArray,sizeof(int)*MAX_NUM_SAMPLE,cudaMemcpyHostToDevice);
        cudaMemcpy(d_sssArray[i].dataNodeIndexArray,h_ssspArray[i]->dataNodeIndexArray,sizeof(int)*MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE,cudaMemcpyHostToDevice);
        cudaMemcpy(d_sssArray[i].dataNodeValueArray,h_ssspArray[i]->dataNodeValueArray,sizeof(float)*MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE,cudaMemcpyHostToDevice);
    }

    //cudaMalloc(&d_sss1,sssSize);
    //cudaMalloc(&d_sss2,sssSize);
    //cudaMalloc(&d_sss3,sssSize);
    cudaMalloc(&d_resultMat,resultSize*sizeof(float));
    cudaMalloc((void**)&d_test_data,test_data_length*sizeof(Data_Node));
    cudaMalloc(&d_test_data_length,sizeof(int));

    recordTime(ev2,ev3,"Mem alloc and copy");
    TIMER_PRINT("malloc",timer);

    TIMER_BEGIN(timer);
    //cudaMemcpy(d_sss1,&sss1,sssSize,cudaMemcpyHostToDevice);
    //cudaMemcpy(d_sss2,&sss2,sssSize,cudaMemcpyHostToDevice);
    //cudaMemcpy(d_sss3,&sss3,sssSize,cudaMemcpyHostToDevice);
    //cudaMemcpy((void*)d_test_data,test_data,test_data_length*sizeof(Data_Node),cudaMemcpyHostToDevice);
    //cudaMemcpy(d_test_data_length,&test_data_length,sizeof(int),cudaMemcpyHostToDevice);
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
    m3gzcKernel<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sss1,d_sss2,d_sss3,d_resultMat);
    //m3gzcKernelWithSharedMemory<<<dimGrid,dimBlock>>>(d_test_data,d_test_data_length,d_sssArray[0],d_sssArray[1],d_sssArray[2],d_resultMat);

    recordTime(ev2,ev3,"GZC computing(kernel 1)");
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
    //cudaMemcpy(resultMat,d_resultMat,resultSize*sizeof(float),cudaMemcpyDeviceToHost);

    TIMER_PRINT("mcopy",timer);

    //cout<<resultMat[0]<<endl;
    //for(int i=0;i<resultSize;i++){
    //    if((i)%sss2.numSample==0 && i>0) cout<<endl;
    //    cout<<resultMat[i]<<'\t';
    //}

    TIMER_BEGIN(timer);
    resultSize=sss1.numSample*sss3.numSample;
    int *resultArray= new int[resultSize];
    int *d_resultArray;
    cudaMalloc(&d_resultArray,resultSize*sizeof(int));

    int threadsPerBlock=128;
    int blockPerGrid=(sss1.numSample-1)/threadsPerBlock+1;

    dimBlock=dim3(BLOCK_SIZE,BLOCK_SIZE);
    dimGrid=dim3((sss1.numSample-1)/BLOCK_SIZE+1,(sss3.numSample-1)/BLOCK_SIZE+1);
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
    recordTime(ev2,ev3,"preprocessing 2");
    //minmaxKernel<<<blockPerGrid,threadsPerBlock>>>(d_resultMat,sss1.numSample,sss2.numSample,sss3.numSample,d_resultArray);
    minmaxKernelImproved<<<dimGrid,dimBlock>>>(d_resultMat,sss1.numSample,sss2.numSample,sss3.numSample,d_resultArray);
    recordTime(ev2,ev3,"min computing (kernel 2)");
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
    cudaMemcpy(resultArray,d_resultArray,resultSize*sizeof(int),cudaMemcpyDeviceToHost);
    TIMER_PRINT("post copy",timer);

    //cout<<sss1.numSample<<' '<<sss2.numSample<<endl;
    //for(int i=0;i<sss1.numSample;i++){
    //    cout<<i<<' '<<resultArray[i]<<endl;
    //}
    //cout<<endl;
    //cout<<resultArray[0]<<endl;

    TIMER_BEGIN(timer);
    
    for(int i=0;i<3;i++){
        cudaFree(d_sssArray[i].labelArray);
        cudaFree(d_sssArray[i].dataNodeOffsetArray);
        cudaFree(d_sssArray[i].dataNodeIndexArray);
        cudaFree(d_sssArray[i].dataNodeValueArray);
    }

    //cudaFree(d_sss1);
    //cudaFree(d_sss2);
    //cudaFree(d_sss3);
    cudaFree(d_resultMat);
    cudaFree(d_test_data);
    cudaFree(d_test_data_length);
    cudaFree(d_resultArray);

    delete test_data;

    recordTime(ev2,ev3,"postprocessing");
    
    cudaEventRecord(ev1,0);
    cudaEventSynchronize(ev0);
    cudaEventSynchronize(ev1);

    cudaEventElapsedTime(&time,ev0,ev1);
    cout<<"cuda event time "<<time<<endl;

    cudaEventDestroy(ev0);
    cudaEventDestroy(ev1);
    cudaEventDestroy(ev2);
    cudaEventDestroy(ev3);

    TIMER_PRINT("post compute",timer);
    TIMER_PRINT("total compute time",timer0);
    delete [] resultMat;
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


int *m3gzcCPU(SerializedSampleSet sss1,SerializedSampleSet sss2,SerializedSampleSet sss3){
    //sss1.print();
    //sss2.print();

    float * sumArray0,* sumArray1,*sumArray2;
    int *resultArray;
    sumArray0=new float[sss2.numSample*sss1.numSample];
    sumArray1=new float[sss3.numSample*sss1.numSample];
    sumArray2=new float[sss3.numSample*sss2.numSample];
    resultArray=new int[sss3.numSample*sss1.numSample];
    
    Data_Node * test_data=new Data_Node;
    test_data->index=1;
    test_data->value=1;

    Data_Sample test_sample;
    test_sample.index=0;
    test_sample.label=0;
    test_sample.data_vector_length=1;
    test_sample.data_vector=test_data;

    //for(int i=0;i<sss1.numSample;i++) sumArray1[i]=getDistance2(test_sample,sss1,i);
    for(int i=0;i<sss1.numSample;i++) 
        for(int j=0;j<sss3.numSample;j++)
            sumArray1[j*sss1.numSample+i]=getDistance2(sss3,j,sss1,i);
    //cout<<sumArray1[0]<<endl;
    //for(int i=0;i<sss2.numSample;i++) sumArray2[i]=getDistance2(test_sample,sss2,i);
    for(int i=0;i<sss2.numSample;i++) 
        for(int j=0;j<sss3.numSample;j++)
            sumArray2[j*sss2.numSample+i]=getDistance2(sss3,j,sss2,i);

    
    for(int i=0;i<sss1.numSample;i++) 
        for(int j=0;j<sss2.numSample;j++)
            sumArray0[i*sss2.numSample+j]=getDistance2(sss2,j,sss1,i);

    for(int i=0;i<sss3.numSample;i++){
        for(int j=0;j<sss1.numSample;j++){
            float min=1;
            for(int k=0;k<sss2.numSample;k++){
                float sum0=sumArray0[j*sss2.numSample+k];
                float v=exp(-4*sumArray1[i*sss1.numSample+j]/sum0)-exp(-4*sumArray2[i*sss2.numSample+k]/sum0);
                if(v<min) min=v;
                //if(v<-THRESHOLD) break;
            }

            if(min>THRESHOLD) resultArray[i*sss1.numSample+j]=1;
            else if(min<-THRESHOLD) resultArray[i*sss1.numSample+j]=-1;
            else resultArray[i*sss1.numSample+j]=0;
        }
    }

    delete test_sample.data_vector;
    return resultArray;
}

int *m3gzcCPUOriginal(SerializedSampleSet sss1,SerializedSampleSet sss2,SerializedSampleSet sss3){
    clock_t timer;
    clock_t timer0;
    float time;
    TIMER_BEGIN(timer);
    TIMER_BEGIN(timer0);

    int * resultArray=new int[sss3.numSample*sss1.numSample];
        
    for(int i=0;i<sss3.numSample;i++){
        for(int j=0;j<sss1.numSample;j++){
            float min=1;
            for(int k=0;k<sss2.numSample;k++){
                float sum0=getDistance2(sss1,j,sss2,k);
                float sum1=getDistance2(sss1,j,sss3,i);
                float sum2=getDistance2(sss2,k,sss3,i);
                //float sum0=sumArray0[j*sss2.numSample+k];
                float v=exp(-4*sum1/sum0)-exp(-4*sum2/sum0);
                if(v<min) min=v;
                //if(v<-THRESHOLD) break;
            }

            if(min>THRESHOLD) resultArray[i*sss1.numSample+j]=1;
            else if(min<-THRESHOLD) resultArray[i*sss1.numSample+j]=-1;
            else resultArray[i*sss1.numSample+j]=0;
        }
    }
    TIMER_PRINT("Execution time",timer);
    return resultArray;
}
