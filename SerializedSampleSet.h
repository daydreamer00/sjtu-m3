#ifndef _SERIALIZEDSAMPLESET_H 
#define _SERIALIZEDSAMPLESET_H

#include<iostream>
#include"util.h"
#define MAX_NUM_SAMPLE 4000
//#define MAX_NUM_DATA 100000

using namespace std;

class SerializedSampleSet {
    private:
        void mallocArray();
    public:
        float *labelArray;//[MAX_NUM_SAMPLE];
        int numSample;
        int *dataNodeOffsetArray;//[MAX_NUM_SAMPLE];
        int *dataNodeIndexArray;//[MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE];
        float *dataNodeValueArray;//[MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE];
        static const int max_num_sample=MAX_NUM_SAMPLE;
        static const int max_num_data=MAX_NUM_SAMPLE*AVERAGE_DATA_PER_SAMPLE;

        SerializedSampleSet(Sample_Link *m_sample_link_head);
        SerializedSampleSet(Sample_Link *m_sample_link_head,bool *enableFlagArray);
        SerializedSampleSet();
        //SerializedSampleSet(const SerializedSampleSet&);
        ~SerializedSampleSet();
        void freeArray();

        void print(){
            cout<<"offset :";
            for(int i=0;i<numSample;i++){
                cout<<dataNodeOffsetArray[i]<<' ';
            }
            cout<<endl<<"index :";
            for(int i=0;i<numSample;i++){
                cout<<dataNodeIndexArray[i]<<' ';
            }
            cout<<endl<<"value :";
            for(int i=0;i<numSample;i++){
                cout<<dataNodeValueArray[i]<<' ';
            }
            cout<<endl;
            cout<<"sample num : "<<numSample;
            cout<<endl;
        }
};

#endif
