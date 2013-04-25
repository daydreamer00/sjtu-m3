#include<iostream>
#include<cuda.h>
#include<stdio.h>
#include"util.h"
#include"SerializedSampleSet.h"
#include"cuPrintf.cu"

#define BLOCK_SIZE 16
#define AVERAGE_DATA_PER_SAMPLE 100
#define SQUARE(x) (x*x)

using namespace std;

__device__ void print(int value){
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==1) 
        printf("block %d,%d, thread %d,%d, value %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
}

__device__ void print(int x,int y,int value){
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==x && threadIdx.y==y) 
        printf("block %d,%d, thread %d,%d, value %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
}

__device__ void print(int x,int y,char * value){
    //if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==x && threadIdx.y==y) 
        printf("block %d,%d, thread %d,%d, value %s\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
}

__device__ void print(float value){
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==1) 
        printf("block %d,%d, thread %d,%d, value %f\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
}


__device__ float getDistance(const Data_Node * data,const int dataLength,const int * dataNodeIndexArray,const float * dataNodeValueArray,const int dataNodeNum){
    float sum=0;
    float x=0,x1=0;
    int i=0,j=0;
    while(1){
        if (i==dataLength){ 
            for(;j<dataNodeNum;j++) sum+=dataNodeValueArray[j]*dataNodeValueArray[j];
            break;
        }
        else if(j==dataNodeNum) {
            for(;i<dataLength;i++) sum+=data[i].value*data[i].value;
            break;
        }
        x=data[i].value;
        x1=dataNodeValueArray[j];
        if(data[i].index<dataNodeIndexArray[j])  {x1=0; i++;}
        else if(data[i].index>dataNodeIndexArray[j]) {x=0; j++;}
        else {i++; j++;};
        float tmp=x-x1;
        sum+=tmp*tmp;
    }
    return sum;
}

__device__ float getDistance(const int * dataNodeIndexArray1,const float * dataNodeValueArray1,const int dataNodeNum1,const int * dataNodeIndexArray2,const float * dataNodeValueArray2,const int dataNodeNum2){
    float x1=0,x2=0;
    float sum=0;
    int loopcount=0;
    //printf("block %d,%d, thread %d,%d, value %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,dataNodeValueArray1[0]);
    //print(dataNodeNum1);
    //print(dataNodeIndexArray1[0]);
    //print(dataNodeValueArray1[0]);
    //print(dataNodeNum2);
    //print(dataNodeIndexArray2[0]);
    //print(dataNodeValueArray2[0]);
    int i=0,j=0;
    while(1){

        //print(i);
        //print(j);
        //print(loopcount);
        loopcount++; //if(loopcount==3) break;
        //if (i==dataNodeNum1){ 
        //    for(;j<dataNodeNum2;j++) sum+=dataNodeValueArray2[j]*dataNodeValueArray2[j];
        //    break;
        //}
        //else if(j==dataNodeNum2) {
        //    for(;i<dataNodeNum1;i++) sum+=dataNodeValueArray1[i]*dataNodeValueArray1[i];
        //    break;
        //}
        //x1=dataNodeValueArray1[i];
        //x2=dataNodeValueArray2[j];

        //int i1=-1,i2=-1;
        //if(i<dataNodeNum1) i1=dataNodeIndexArray1[i];
        //if(j<dataNodeNum2) i2=dataNodeIndexArray2[j];

        if (i==dataNodeNum1 && j==dataNodeNum2) {break;}
        else if((i<dataNodeNum1 && j==dataNodeNum2) || (j<dataNodeNum2 && i<dataNodeNum1 && dataNodeIndexArray1[i]<dataNodeIndexArray2[j]))  {
            x1=dataNodeValueArray1[j];
            x2=0; 
            i++;
        } else if((i==dataNodeNum1 && j<dataNodeNum2) || 
                (j<dataNodeNum2 && i<dataNodeNum1 && dataNodeIndexArray1[i]>dataNodeIndexArray2[j]) ){
            x2=dataNodeValueArray2[j];
            x1=0;
            j++;
        } else {
            x1=dataNodeValueArray1[i];
            x2=dataNodeValueArray2[j];
            i++; j++;
        }
        float tmp=x1-x2;
        sum+=tmp*tmp;
    }
    //print(sum);
    return sum;
}


__device__ void loadToSharedMemory(int ix0,int ix1,int iy0,int iy1,
        int xbegin,int xend,int ybegin,int yend,
        int ix,int iy,const SerializedSampleSet *sss1,const SerializedSampleSet *sss2,
        int *dataOffsetArray1,int *dataIndexArray1,float *dataValueArray1,
        int *dataOffsetArray2,int *dataIndexArray2,float *dataValueArray2){

    if(threadIdx.y==blockDim.y-1) dataOffsetArray1[threadIdx.x]=sss1->dataNodeOffsetArray[ix];
    if(threadIdx.x==blockDim.x-1) dataOffsetArray2[threadIdx.y]=sss2->dataNodeOffsetArray[iy];

    //These variables should be in shared memory
    //int ix0,iy0,ix1,iy1;
    //int xbegin,ybegin,xend,yend;

    //if(threadIdx.x==0 && threadIdx.y==0){
    //This block of code needs only execute once on a block
        //int ix0=blockIdx.x*blockDim.x,iy0=blockIdx.y*blockIdx.y;
        //int ix1=blockIdx.x<gridDim.x-1?ix0+blockDim.x-1:sss1->numSample-1;
        //int iy1=blockIdx.y<gridDim.y-1?iy0+blockDim.y-1:sss2->numSample-1;
        //int xbegin=ix0>0?sss1->dataNodeOffsetArray[ix0-1]:0;
        //int ybegin=iy0>0?sss2->dataNodeOffsetArray[iy0-1]:0;
        //int xend=sss1->dataNodeOffsetArray[ix1];
        //int yend=sss2->dataNodeOffsetArray[iy1];
    //}

    //print(xend);

    if(xend-xbegin>BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE || yend-ybegin>BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE)
        print(0,0,"too many data");

    if(xend-xbegin>0) 
        for(int i=0;i<(xend-xbegin-1)/(blockDim.x*blockDim.y)+1;i++) {
            int j=i*blockDim.x*blockDim.y+threadIdx.x*blockDim.y+threadIdx.y;
            if(j+xbegin<xend) {
                dataIndexArray1[j]=sss1->dataNodeIndexArray[xbegin+j];
                dataValueArray1[j]=sss1->dataNodeValueArray[xbegin+j];
            }
        }
    if(yend-ybegin>0) 
        for(int i=0;i<(yend-ybegin-1)/(blockDim.x*blockDim.y)+1;i++) {
            int j=ybegin+i*blockDim.x*blockDim.y+threadIdx.x*blockDim.y+threadIdx.y;
            if(j+ybegin<yend) {
                dataIndexArray2[j]=sss2->dataNodeIndexArray[ybegin+j];
                dataValueArray2[j]=sss2->dataNodeValueArray[ybegin+j];
            }
        }
}


__global__ void m3gzcKernel(const Data_Node * data,const int * dataLength,const SerializedSampleSet *sss1,const SerializedSampleSet *sss2,float * resultMat){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    float x=0,x1=0,x2=0,tmp=0;

    if(ix>=sss1->numSample) return;
    if(iy>=sss2->numSample) return;
    int xbegin=ix>0?sss1->dataNodeOffsetArray[ix-1]:0;
    int ybegin=iy>0?sss2->dataNodeOffsetArray[iy-1]:0;
    int xend=sss1->dataNodeOffsetArray[ix];
    int yend=sss2->dataNodeOffsetArray[iy];

    float sum1=0,sum2=0,sum0=0;
    
    //print(ix);
    //print(iy);
    //cuPrintf("%d\n",3);
    //print(xbegin);
    //print(sss1->dataNodeValueArray[xbegin]);
    //print(ybegin);
    //print(sss2->dataNodeValueArray[ybegin]);
    //print(sss2->dataNodeValueArray[ybegin+1]);
    //print(sss2->dataNodeValueArray[ybegin+2]);
    sum0=getDistance(&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
    sum1=getDistance(data,*dataLength,&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin);
    sum2=getDistance(data,*dataLength,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
    //print(sum0);
    //print(sum1);
    //print(sum2);

    float theta2=0.25*sum0;

    float result=expf(-sum1/theta2)-expf(-sum2/theta2);

    //print(result);

    resultMat[ix*(sss2->numSample)+iy]=result;

    return;
}

__global__ void m3gzcKernelWithSharedMemory(const Data_Node * data,const int * dataLength,const SerializedSampleSet *sss1,const SerializedSampleSet *sss2,float * resultMat){

    __shared__ float sum1[BLOCK_SIZE];
    __shared__ float sum2[BLOCK_SIZE];
    __shared__ int dataOffsetArray1[BLOCK_SIZE]; 
    __shared__ int dataOffsetArray2[BLOCK_SIZE]; 
    __shared__ int dataIndexArray1[BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE]; 
    __shared__ int dataIndexArray2[BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE]; 
    __shared__ float dataValueArray1[BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE]; 
    __shared__ float dataValueArray2[BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE]; 

    __shared__ int ixBlockBegin,ixBlockEnd,iyBlockBegin,iyBlockEnd;
    __shared__ int xBlockBegin,xBlockEnd,yBlockBegin,yBlockEnd;

    //__shared__ float dataValueArray3[1200000]; 
    //dataValueArray3[1199999]=0;
    
    if(threadIdx.x==0 && threadIdx.y==0){
        ixBlockBegin=blockIdx.x*blockDim.x;
        iyBlockBegin=blockIdx.y*blockIdx.y;
        ixBlockEnd=blockIdx.x<gridDim.x-1?ixBlockBegin+blockDim.x-1:sss1->numSample-1;
        iyBlockEnd=blockIdx.y<gridDim.y-1?iyBlockBegin+blockDim.y-1:sss2->numSample-1;
        xBlockBegin=ixBlockBegin>0?sss1->dataNodeOffsetArray[ixBlockBegin-1]:0;
        yBlockBegin=iyBlockBegin>0?sss2->dataNodeOffsetArray[iyBlockBegin-1]:0;
        xBlockEnd=sss1->dataNodeOffsetArray[ixBlockEnd];
        yBlockEnd=sss2->dataNodeOffsetArray[iyBlockEnd];
    }

    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    float x=0,x1=0,x2=0,tmp=0;

    loadToSharedMemory(ixBlockBegin,ixBlockEnd,iyBlockBegin,iyBlockEnd,
            xBlockBegin,xBlockEnd,yBlockBegin,yBlockEnd,
            ix,iy,sss1,sss2,
            dataOffsetArray1,dataIndexArray1,dataValueArray1,
            dataOffsetArray2,dataIndexArray2,dataValueArray2);

    __syncthreads();
    //for(int i=0;i<BLOCK_SIZE;i++) print(dataIndexArray2[i]);

    if(ix>=sss1->numSample) return;
    if(iy>=sss2->numSample) return;

    int xbegin=ix>0?dataOffsetArray1[ix-1]:0;
    int ybegin=iy>0?dataOffsetArray2[iy-1]:0;
    int xend=dataOffsetArray1[ix];
    int yend=dataOffsetArray2[iy];

    if(threadIdx.y==0)
        sum1[threadIdx.x]=getDistance(data,*dataLength,&(dataIndexArray1[xbegin]),&(dataValueArray1[xbegin]),xend-xbegin);
    if(threadIdx.x==0)
        sum2[threadIdx.y]=getDistance(data,*dataLength,&(dataIndexArray2[ybegin]),&(dataValueArray2[ybegin]),yend-ybegin);

    __syncthreads();

    float sum0=0;
    //float sum1=0,sum2=0;
    
    //print(ix);
    //print(iy);
    //cuPrintf("%d\n",3);
    //print(xbegin);
    //print(sss1->dataNodeValueArray[xbegin]);
    //print(ybegin);
    //print(sss2->dataNodeValueArray[ybegin]);
    //print(sss2->dataNodeValueArray[ybegin+1]);
    //print(sss2->dataNodeValueArray[ybegin+2]);

    sum0=getDistance(&(dataIndexArray1[xbegin]),&(dataValueArray1[xbegin]),xend-xbegin,&(dataIndexArray2[ybegin]),&(dataValueArray2[ybegin]),yend-ybegin);
    
    //print(sum0);
    //print(sum1);
    //print(sum2);

    float theta2=0.25*sum0;

    float result;
    result=expf(-sum1[threadIdx.x]/theta2)-expf(-sum2[threadIdx.y]/theta2);

    //print(result);

    resultMat[ix*(sss2->numSample)+iy]=result;

    return;
}
