#include<iostream>
#include<cuda.h>
#include"util.h"
#include"SerializedSampleSet.h"
#include"cuPrintf.cu"

using namespace std;

__device__ float getDistance(const Data_Node * data,const int dataLength,const int * dataNodeIndexArray,const float * dataNodeValueArray,const int dataNodeNum){
    float sum=0;
    float x=0,x1=0;
    while(1){
        int i=0,j=0;
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
        float tmp=x-x1;
        sum+=tmp*tmp;
    }
    return sum;
}

__device__ float getDistance(const int * dataNodeIndexArray1,const float * dataNodeValueArray1,const int dataNodeNum1,const int * dataNodeIndexArray2,const float * dataNodeValueArray2,const int dataNodeNum2){
    float x1=0,x2=0;
    float sum=0;
    cuPrintf("%d\n",dataNodeNum1);
    while(1){
        int i=0,j=0;
        if (i==dataNodeNum1){ 
            for(;j<dataNodeNum2;j++) sum+=dataNodeValueArray2[j]*dataNodeValueArray2[j];
            break;
        }
        else if(j==dataNodeNum2) {
            for(;i<dataNodeNum1;i++) sum+=dataNodeValueArray1[i]*dataNodeValueArray1[i];
            break;
        }
        x1=dataNodeValueArray1[j];
        x2=dataNodeValueArray2[j];
        if(dataNodeIndexArray1[i]<dataNodeIndexArray2[j])  {x2=0; i++;}
        else if(dataNodeIndexArray1[i]>dataNodeIndexArray2[j]) {x1=0; j++;}
        float tmp=x1-x2;
        sum+=tmp*tmp;
    }
    return sum;
}


__global__ void m3gzcKernel(const Data_Node * data,const int * dataLength,const SerializedSampleSet *sss1,const SerializedSampleSet *sss2,int * resultMat){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    float x=0,x1=0,x2=0,tmp=0;

    int lx=0,ly=0;
    if(ix>=sss1->numSample) return;
    if(iy>=sss2->numSample) return;
    int xbegin=ix>0?sss1->dataNodeOffsetArray[ix-1]:0;
    int ybegin=ix>0?sss2->dataNodeOffsetArray[iy-1]:0;
    int xend=sss1->dataNodeOffsetArray[ix];
    int yend=sss2->dataNodeOffsetArray[iy];

    float sum1=0,sum2=0,sum0=0;
    
    sum0=getDistance(&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
    //sum1=getDistance(data,*dataLength,&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin);
    //sum2=getDistance(data,*dataLength,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);

    float theta2=0.25*sum0;
    

    float result=exp10f(-sum1/theta2)-exp10f(-sum2/theta2);

    resultMat[ix*(SerializedSampleSet::max_num_sample)+iy]=result;

    return;
}
