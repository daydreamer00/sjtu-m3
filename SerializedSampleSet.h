#ifndef _SERIALIZEDSAMPLESET_H 
#define _SERIALIZEDSAMPLESET_H

#include<iostream>
#include"util.h"
#define MAX_NUM_SAMPLE 1000
#define MAX_NUM_DATA 100000

using namespace std;

class SerializedSampleSet {
    public:
        float labelArray[MAX_NUM_SAMPLE];
        int numSample;
        int dataNodeOffsetArray[MAX_NUM_SAMPLE];
        int dataNodeIndexArray[MAX_NUM_DATA];
        float dataNodeValueArray[MAX_NUM_DATA];
        static const int max_num_sample=MAX_NUM_SAMPLE;
        static const int max_num_data=MAX_NUM_DATA;

        SerializedSampleSet(Sample_Link *m_sample_link_head);
        SerializedSampleSet(){
            memset(labelArray,0,sizeof(labelArray));
            numSample=0;
            memset(dataNodeOffsetArray,0,sizeof(dataNodeOffsetArray));
            memset(dataNodeIndexArray,0,sizeof(dataNodeIndexArray));
            memset(dataNodeValueArray,0,sizeof(dataNodeValueArray));
        };

        void print(){
            for(int i=0;i<20;i++){
                //cout<<dataNodeValueArray[i];
            }
            cout<<numSample;
            cout<<endl;
        }
};

#endif
