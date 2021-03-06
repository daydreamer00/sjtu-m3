#include<iostream>
#include<cuda.h>
#include<stdio.h>
#include<cuda_runtime.h>
//#include<helpler_functions.h>

#include"util.h"
#include"SerializedSampleSet.h"
#include"cuPrintf.cu"


using namespace std;

__device__ void print(int value){
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0) 
    //if(blockIdx.x==0 && threadIdx.x<10)
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
    if(blockIdx.x==0 && blockIdx.y==0 && threadIdx.x==0 && threadIdx.y==0) 
        printf("block %d,%d, thread %d,%d, value %f\n",blockIdx.x,blockIdx.y,threadIdx.x,threadIdx.y,value);
}


__device__ float getDistance(const Data_Node * data,const int dataLength,const int * dataNodeIndexArray,const float * dataNodeValueArray,const int dataNodeNum){
    float sum=0;
    float x=0,x1=0;
    int i=0,j=0;
    while(1){
        if (i==dataLength && j==dataNodeNum) {break;}
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

__device__ float getDistance_alt2(const int * dataNodeIndexArray1,const float * dataNodeValueArray1,const int dataNodeNum1,const int * dataNodeIndexArray2,const float * dataNodeValueArray2,const int dataNodeNum2){
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
        if (i==dataNodeNum1){ 
            for(;j<dataNodeNum2;j++) sum+=dataNodeValueArray2[j]*dataNodeValueArray2[j];
            break;
        }
        else if(j==dataNodeNum2) {
            for(;i<dataNodeNum1;i++) sum+=dataNodeValueArray1[i]*dataNodeValueArray1[i];
            break;
        }
        x1=0;
        x2=0;
        if(dataNodeIndexArray1[i]==loopcount) {
            x1=dataNodeValueArray1[i];
            i++;
        }
        if(dataNodeIndexArray2[j]==loopcount){
            x2=dataNodeValueArray2[j];
            j++;
        }

        //if (i==dataNodeNum1 && j==dataNodeNum2) {break;}
        //else if((i<dataNodeNum1 && j==dataNodeNum2) || (j<dataNodeNum2 && i<dataNodeNum1 && dataNodeIndexArray1[i]<dataNodeIndexArray2[j]))  {
        //    x1=dataNodeValueArray1[j];
        //    x2=0; 
        //    i++;
        //} else if((i==dataNodeNum1 && j<dataNodeNum2) || 
        //        (j<dataNodeNum2 && i<dataNodeNum1 && dataNodeIndexArray1[i]>dataNodeIndexArray2[j]) ){
        //    x2=dataNodeValueArray2[j];
        //    x1=0;
        //    j++;
        //} else {
        //    x1=dataNodeValueArray1[i];
        //    x2=dataNodeValueArray2[j];
        //    i++; j++;
        //}
        float temp=x1-x2;
        sum+=temp*temp;
        loopcount++; //if(loopcount==3) break;
    }
    //print(sum);
    return sum;
}

__device__ float getDistance_alt1(const int * dataNodeIndexArray1,const float * dataNodeValueArray1,const int dataNodeNum1,const int * dataNodeIndexArray2,const float * dataNodeValueArray2,const int dataNodeNum2){
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
        if (i==dataNodeNum1){ 
            for(;j<dataNodeNum2;j++) sum+=dataNodeValueArray2[j]*dataNodeValueArray2[j];
            break;
        }
        else if(j==dataNodeNum2) {
            for(;i<dataNodeNum1;i++) sum+=dataNodeValueArray1[i]*dataNodeValueArray1[i];
            break;
        }
        x1=dataNodeValueArray1[i];
        x2=dataNodeValueArray2[j];
        if(dataNodeIndexArray1[i]<dataNodeIndexArray2[j]) {
            x2=0;
            i++;
        }else if(dataNodeIndexArray1[i]>dataNodeIndexArray2[j]){
            x1=0;
            j++;
        }else{
            i++;
            j++;
        }

        //if (i==dataNodeNum1 && j==dataNodeNum2) {break;}
        //else if((i<dataNodeNum1 && j==dataNodeNum2) || (j<dataNodeNum2 && i<dataNodeNum1 && dataNodeIndexArray1[i]<dataNodeIndexArray2[j]))  {
        //    x1=dataNodeValueArray1[j];
        //    x2=0; 
        //    i++;
        //} else if((i==dataNodeNum1 && j<dataNodeNum2) || 
        //        (j<dataNodeNum2 && i<dataNodeNum1 && dataNodeIndexArray1[i]>dataNodeIndexArray2[j]) ){
        //    x2=dataNodeValueArray2[j];
        //    x1=0;
        //    j++;
        //} else {
        //    x1=dataNodeValueArray1[i];
        //    x2=dataNodeValueArray2[j];
        //    i++; j++;
        //}
        float temp=x1-x2;
        sum+=temp*temp;
    }
    //print(sum);
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
        float temp=x1-x2;
        sum+=temp*temp;
    }
    //print(sum);
    return sum;
}


__device__ void loadToSharedMemory(int ithread,const SerializedSampleSet *sss1,int iBlockBegin1,int iBlockEnd1,int idataBlockBegin1,int idataBlockEnd1,const SerializedSampleSet *sss2,int iBlockBegin2,int iBlockEnd2,int idataBlockBegin2,int idataBlockEnd2,float * sumArray){
       //int *dataOffsetArray1,int *dataIndexArray1,float *dataValueArray1,
       //int *dataOffsetArray2,int *dataIndexArray2,float *dataValueArray2){

    int width=iBlockEnd1-iBlockBegin1+1,length=iBlockEnd2-iBlockBegin2+1;

    int maxi=(width*length-1)/(BLOCK_SIZE*BLOCK_SIZE)+1;

    for(int i=0;i<maxi;i++){
        int it=i*(SQUARE(BLOCK_SIZE))+ithread;
        if (it>=width*length) return;
        int ix=iBlockBegin1+it/length,iy=iBlockBegin2+it%length;
        int xbegin=ix>0?sss1->dataNodeOffsetArray[ix-1]:0;
        int ybegin=iy>0?sss2->dataNodeOffsetArray[iy-1]:0;
        int xend=sss1->dataNodeOffsetArray[ix];
        int yend=sss2->dataNodeOffsetArray[iy];
        sumArray[it]=getDistance_alt1(&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
        //sumArray[it]=getDistance(&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
    }

    ////if(threadIdx.x==0 && threadIdx.y==0) {
    //    xOffsetBias=blockIdx.x==0?0:sss1->dataNodeOffsetArray[ix0-1];
    //    yOffsetBias=blockIdx.y==0?0:sss2->dataNodeOffsetArray[iy0-1];
    ////}
    //    //print(iy0);
    //    //print(sss2->dataNodeOffsetArray[iy0-1]);

    //if(threadIdx.y==blockDim.y-1) {
    //    dataOffsetArray1[threadIdx.x]=sss1->dataNodeOffsetArray[ix]-xOffsetBias;
    //    //print(ix);
    //}
    //if(threadIdx.x==blockDim.x-1) dataOffsetArray2[threadIdx.y]=sss2->dataNodeOffsetArray[iy]-yOffsetBias;

    ////These variables should be in shared memory
    ////int ix0,iy0,ix1,iy1;
    ////int xbegin,ybegin,xend,yend;

    ////if(threadIdx.x==0 && threadIdx.y==0){
    ////This block of code needs only execute once on a block
    //    //int ix0=blockIdx.x*blockDim.x,iy0=blockIdx.y*blockIdx.y;
    //    //int ix1=blockIdx.x<gridDim.x-1?ix0+blockDim.x-1:sss1->numSample-1;
    //    //int iy1=blockIdx.y<gridDim.y-1?iy0+blockDim.y-1:sss2->numSample-1;
    //    //int xbegin=ix0>0?sss1->dataNodeOffsetArray[ix0-1]:0;
    //    //int ybegin=iy0>0?sss2->dataNodeOffsetArray[iy0-1]:0;
    //    //int xend=sss1->dataNodeOffsetArray[ix1];
    //    //int yend=sss2->dataNodeOffsetArray[iy1];
    ////}

    ////print(xend);

    //if(xend-xbegin>BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE || yend-ybegin>BLOCK_SIZE*AVERAGE_DATA_PER_SAMPLE)
    //    print(0,0,"too many data");

    //if(xend-xbegin>0) 
    //    for(int i=0;i<(xend-xbegin-1)/(blockDim.x*blockDim.y)+1;i++) {
    //        int j=i*blockDim.x*blockDim.y+threadIdx.x*blockDim.y+threadIdx.y;
    //        if(j+xbegin<xend) {
    //            dataIndexArray1[j]=sss1->dataNodeIndexArray[xbegin+j];
    //            dataValueArray1[j]=sss1->dataNodeValueArray[xbegin+j];
    //        }
    //    }
    ////print(yend);
    //if(yend-ybegin>0) 
    //    for(int i=0;i<(yend-ybegin-1)/(blockDim.x*blockDim.y)+1;i++) {
    //        int j=i*blockDim.x*blockDim.y+threadIdx.x*blockDim.y+threadIdx.y;
    //        if(j+ybegin<yend) {
    //            dataIndexArray2[j]=sss2->dataNodeIndexArray[ybegin+j];
    //            dataValueArray2[j]=sss2->dataNodeValueArray[ybegin+j];
    //            //print(dataValueArray2[j]);
    //        }
    //    }
}


__global__ void m3gzcKernel(const Data_Node * data,const int * dataLength,const SerializedSampleSet sss1,const SerializedSampleSet sss2,const SerializedSampleSet sss3,float * resultMat){
    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    float x=0,x1=0,x2=0,tmp=0;

    if(ix>=sss1.numSample) return;
    if(iy>=sss2.numSample) return;
    int xbegin=ix>0?sss1.dataNodeOffsetArray[ix-1]:0;
    int ybegin=iy>0?sss2.dataNodeOffsetArray[iy-1]:0;
    int xend=sss1.dataNodeOffsetArray[ix];
    int yend=sss2.dataNodeOffsetArray[iy];

    int iz=0;

    while(iz<sss3.numSample){
        int zbegin=iz>0?sss3.dataNodeOffsetArray[iz-1]:0;
        int zend=sss3.dataNodeOffsetArray[iz];

        float sum1=0,sum2=0,sum0=0;

        //print(ix);
        //print(iy);
        //print(xbegin);
        //print(sss1.dataNodeValueArray[xbegin]);
        //print(ybegin);
        //print(sss2.dataNodeValueArray[ybegin]);
        //print(sss2.dataNodeValueArray[ybegin+1]);
        //print(sss2.dataNodeValueArray[ybegin+2]);
        sum0=getDistance(&(sss1.dataNodeIndexArray[xbegin]),&(sss1.dataNodeValueArray[xbegin]),xend-xbegin,&(sss2.dataNodeIndexArray[ybegin]),&(sss2.dataNodeValueArray[ybegin]),yend-ybegin);
        sum1=getDistance(&(sss3.dataNodeIndexArray[zbegin]),&(sss3.dataNodeValueArray[zbegin]),zend-zbegin,&(sss1.dataNodeIndexArray[xbegin]),&(sss1.dataNodeValueArray[xbegin]),xend-xbegin);
        sum2=getDistance(&(sss3.dataNodeIndexArray[zbegin]),&(sss3.dataNodeValueArray[zbegin]),zend-zbegin,&(sss2.dataNodeIndexArray[ybegin]),&(sss2.dataNodeValueArray[ybegin]),yend-ybegin);
        //sum1=getDistance(data,*dataLength,&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin);
        //sum2=getDistance(data,*dataLength,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
        //if(ix==24 && iy==0) print(sum0);
        //if(ix==24 && iy==0) print(sum1);
        //if(ix==24 && iy==0) print(sum2);

        float theta2=0.25*sum0;

        float result=__expf(-sum1/theta2)-expf(-sum2/theta2);
        //if(ix==24) print(result);

        //print(result);

        resultMat[iz*(sss1.numSample)*(sss2.numSample)+ix*(sss2.numSample)+iy]=result;
        iz++;
    }

    return;
}

__global__ void m3gzcKernelWithSharedMemory(const Data_Node * data,const int * dataLength,const SerializedSampleSet sss1,const SerializedSampleSet sss2,const SerializedSampleSet sss3,float * resultMat){

    __shared__ float sum0[BLOCK_SIZE*BLOCK_SIZE]; 
    __shared__ float sum1[BLOCK_SIZE*TEST_SHARD_SIZE]; 
    __shared__ float sum2[BLOCK_SIZE*TEST_SHARD_SIZE]; 

    //__shared__ int ixBlockBegin,ixBlockEnd,iyBlockBegin,iyBlockEnd;
    //__shared__ int xBlockBegin,xBlockEnd,yBlockBegin,yBlockEnd;
    __shared__ int iBlockBegin[3],iBlockEnd[3],idataBlockBegin[3],idataBlockEnd[3];

    //__shared__ float dataValueArray3[1200000]; 
    //dataValueArray3[1199999]=0;
    
    if(threadIdx.x==0 && threadIdx.y==0){
        int i=threadIdx.x;
        iBlockBegin[0]=blockIdx.x*blockDim.x;
        iBlockBegin[1]=blockIdx.y*blockDim.y;
        iBlockBegin[2]=0;
        iBlockEnd[0]=blockIdx.x<gridDim.x-1?iBlockBegin[0]+blockDim.x-1:sss1.numSample-1;
        iBlockEnd[1]=blockIdx.y<gridDim.y-1?iBlockBegin[1]+blockDim.y-1:sss2.numSample-1;
        iBlockEnd[2]=sss3.numSample-1;
        idataBlockBegin[0]=iBlockBegin[0]>0?sss1.dataNodeOffsetArray[iBlockBegin[0]-1]:0;
        idataBlockBegin[1]=iBlockBegin[1]>0?sss2.dataNodeOffsetArray[iBlockBegin[1]-1]:0;
        idataBlockBegin[2]=0;
        idataBlockEnd[0]=sss1.dataNodeOffsetArray[iBlockEnd[0]]-1;
        idataBlockEnd[1]=sss2.dataNodeOffsetArray[iBlockEnd[1]]-1;
        idataBlockEnd[2]=sss2.dataNodeOffsetArray[iBlockEnd[2]]-1;

        //ixBlockBegin=blockIdx.x*blockDim.x;
        //iyBlockBegin=blockIdx.y*blockDim.y;
        //ixBlockEnd=blockIdx.x<gridDim.x-1?ixBlockBegin+blockDim.x-1:sss1->numSample-1;
        //iyBlockEnd=blockIdx.y<gridDim.y-1?iyBlockBegin+blockDim.y-1:sss2->numSample-1;
        //xBlockBegin=ixBlockBegin>0?sss1->dataNodeOffsetArray[ixBlockBegin-1]:0;
        //yBlockBegin=iyBlockBegin>0?sss2->dataNodeOffsetArray[iyBlockBegin-1]:0;
        //xBlockEnd=sss1->dataNodeOffsetArray[ixBlockEnd];
        //yBlockEnd=sss2->dataNodeOffsetArray[iyBlockEnd];
    }
    __syncthreads();

    int ix=blockIdx.x*blockDim.x+threadIdx.x;
    int iy=blockIdx.y*blockDim.y+threadIdx.y;
    float x=0;//,x1=0,x2=0,tmp=0;
    int ithread=threadIdx.x*blockDim.y+threadIdx.y;

    //print(yBlockBegin);
    //print(yBlockEnd);
    //print(iyBlockEnd);

    loadToSharedMemory(ithread,&sss1,iBlockBegin[0],iBlockEnd[0],idataBlockBegin[0],idataBlockEnd[0],&sss2,iBlockBegin[1],iBlockEnd[1],idataBlockBegin[1],idataBlockEnd[1],sum0);
    loadToSharedMemory(ithread,&sss3,iBlockBegin[2],iBlockEnd[2],idataBlockBegin[2],idataBlockEnd[2],&sss1,iBlockBegin[0],iBlockEnd[0],idataBlockBegin[0],idataBlockEnd[0],sum1);
    loadToSharedMemory(ithread,&sss3,iBlockBegin[2],iBlockEnd[2],idataBlockBegin[2],idataBlockEnd[2],&sss2,iBlockBegin[1],iBlockEnd[1],idataBlockBegin[1],idataBlockEnd[1],sum2);

    //        ixBlockBegin,ixBlockEnd,iyBlockBegin,iyBlockEnd,
    //        xBlockBegin,xBlockEnd,yBlockBegin,yBlockEnd,
    //        ix,iy,sss1,sss2,
    //        dataOffsetArray1,dataIndexArray1,dataValueArray1,
    //        dataOffsetArray2,dataIndexArray2,dataValueArray2);

    __syncthreads();
    //for(int i=0;i<BLOCK_SIZE;i++) print(dataIndexArray2[i]);

    if(ix>=sss1.numSample) return;
    if(iy>=sss2.numSample) return;

    int iz=0;

    while(iz<sss3.numSample){
        //int zbegin=iz>0?sss3->dataNodeOffsetArray[iz-1]:0;
        //int zend=sss3->dataNodeOffsetArray[iz];

        //float sum1=,sum2=0,sum0=0;

        //print(ix);
        //print(iy);
        //print(xbegin);
        //print(sss1->dataNodeValueArray[xbegin]);
        //print(ybegin);
        //print(sss2->dataNodeValueArray[ybegin]);
        //print(sss2->dataNodeValueArray[ybegin+1]);
        //print(sss2->dataNodeValueArray[ybegin+2]);
        //sum0=getDistance(&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);
        //sum1=getDistance(&(sss3->dataNodeIndexArray[zbegin]),&(sss3->dataNodeValueArray[zbegin]),zend-zbegin,&(sss1->dataNodeIndexArray[xbegin]),&(sss1->dataNodeValueArray[xbegin]),xend-xbegin);
        //sum2=getDistance(&(sss3->dataNodeIndexArray[zbegin]),&(sss3->dataNodeValueArray[zbegin]),zend-zbegin,&(sss2->dataNodeIndexArray[ybegin]),&(sss2->dataNodeValueArray[ybegin]),yend-ybegin);

        float theta2=0.25*sum0[threadIdx.x*(iBlockEnd[1]-iBlockBegin[1]+1)+threadIdx.y];

        float result=expf(-sum1[iz*(iBlockEnd[0]-iBlockBegin[0]+1)+threadIdx.x]/theta2)-expf(-sum2[iz*(iBlockEnd[1]-iBlockBegin[1]+1)+threadIdx.y]/theta2);
        //if(ix==24) print(result);

        //print(result);

        resultMat[iz*(sss1.numSample)*(sss2.numSample)+ix*(sss2.numSample)+iy]=result;
        iz++;
    }

    //int xbegin=threadIdx.x>0?dataOffsetArray1[threadIdx.x-1]:0;
    //int ybegin=threadIdx.y>0?dataOffsetArray2[threadIdx.y-1]:0;
    //int xend=dataOffsetArray1[threadIdx.x];
    //int yend=dataOffsetArray2[threadIdx.y];

    //if(threadIdx.y==0)
    //    sum1[threadIdx.x]=getDistance(data,*dataLength,&(dataIndexArray1[xbegin]),&(dataValueArray1[xbegin]),xend-xbegin);
    //if(threadIdx.x==0)
    //    sum2[threadIdx.y]=getDistance(data,*dataLength,&(dataIndexArray2[ybegin]),&(dataValueArray2[ybegin]),yend-ybegin);

    //__syncthreads();

    //float sum0=0;
    //float sum1=0,sum2=0;
    
    //print(ix);
    //print(iy);
    //cuPrintf("%d\n",3);
    //print(xbegin);
    //print(xend);
    //print(sss1->dataNodeValueArray[xbegin]);
    //print(ybegin);
    //print(yend);
    //print(sss2->dataNodeValueArray[ybegin]);
    //print(sss2->dataNodeValueArray[ybegin+1]);
    //print(sss2->dataNodeValueArray[ybegin+2]);
    //print(dataValueArray1[xbegin]);
    //print(dataValueArray2[ybegin]);

    //sum0=getDistance(&(dataIndexArray1[xbegin]),&(dataValueArray1[xbegin]),xend-xbegin,&(dataIndexArray2[ybegin]),&(dataValueArray2[ybegin]),yend-ybegin);
    
    //print(sum0);
    //print(sum1);
    //print(sum2);

    //float theta2=0.25*sum0;

    //float result;
    //result=expf(-sum1[threadIdx.x]/theta2)-expf(-sum2[threadIdx.y]/theta2);

    ////print(result);

    //resultMat[ix*(sss2->numSample)+iy]=result;

    return;
}

__global__ void minmaxKernel(float *resultMat,int width,int length,int height,int *resultArray){
    //__shared__ int sharedWidth,sharedLength;
    int sharedWidth=width,sharedLength=length;
    //if(blockIdx.x==0 && threadIdx.x==0) {
    //    sharedWidth=width;
    //    sharedLength=length;
    //}
    //__syncthreads();
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    if(x>=sharedWidth) {
        return;
    }

    for(int z=0;z<height;z++) {
        int i=0;
        float min=1;
        for(int i=0;i<sharedLength;i++){
        //while(1){
            //if(i==sharedLength) break;
            float tmpValue=resultMat[z*width*length+x*sharedLength+i];
            if(tmpValue<min) min=tmpValue;
            //i++;
            //if(min<-THRESHOLD) break;
        //}
        }
        if(min>THRESHOLD) resultArray[z*width+x]=1;
        else if (min<-THRESHOLD) resultArray[z*width+x]=-1;
        else resultArray[z*width+x]=0;
    }
    //print(resultArray[x]);
    return;
}

__global__ void minmaxKernelImproved(float *resultMat,int width,int length,int height,int *resultArray){
    int x=blockIdx.x*blockDim.x+threadIdx.x;
    int z=blockIdx.y*blockDim.y+threadIdx.y;
    if(x>=width) return;
    if(z>=height) return;
    float min=1;
    for(int i=0;i<length;i++){
        //while(1){
        //if(i==sharedLength) break;
        float tmpValue=resultMat[z*width*length+x*length+i];
        if(tmpValue<min) min=tmpValue;
        //i++;
        //if(min<-THRESHOLD) break;
        //}
    }
    if(min>THRESHOLD) resultArray[z*width+x]=1;
    else if (min<-THRESHOLD) resultArray[z*width+x]=-1;
    else resultArray[z*width+x]=0;
    return;
}
