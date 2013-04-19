#include<iostream>
#include<cuda.h>
#include<stdio.h>
#include"util.h"
#include"SerializedSampleSet.h"
#include"cuPrintf.cu"

using namespace std;

__device__ void print(int value){
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==1) 
        printf("block %d,%d, thread %d,%d, value %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
}

__device__ void print(int x,int y,int value){
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==x && threadIdx.y==y) 
        printf("block %d,%d, thread %d,%d, value %d\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
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


__global__ void m3gzcKernel(const Data_Node * data,const int * dataLength,const SerializedSampleSet *sss1,const SerializedSampleSet *sss2,float * resultMat){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    float x=0,x1=0,x2=0,tmp=0;

    int lx=0,ly=0;
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
