#include"SerializedSampleSet.h"

using namespace std;

SerializedSampleSet::SerializedSampleSet(Sample_Link * m_sample_link_head){
    SerializedSampleSet();
    Sample_Link * cur_sample_link=m_sample_link_head;
    Data_Sample * cur_sample=NULL;
    Data_Node * cur_data_vector=NULL;
    int i=0,dataNodeOffset=0;
    while(cur_sample_link){
        if(i>=SerializedSampleSet::max_num_sample) error("Too many samples"); 
        cur_sample=cur_sample_link->sample_head;
        cur_data_vector=cur_sample->data_vector;
        labelArray[i]=cur_sample->label;
        for(int j=0;j<cur_sample->data_vector_length;j++){
            if(dataNodeOffset+j>=SerializedSampleSet::max_num_data)  error("Too many data nodes");
            //cout<<cur_data_vector[j].index<<" ";
            dataNodeIndexArray[dataNodeOffset+j]=cur_data_vector[j].index;
            dataNodeValueArray[dataNodeOffset+j]=cur_data_vector[j].value;
            //cout<<dataNodeValueArray[dataNodeOffset+j]<<endl;
        }
        dataNodeOffset+=cur_sample->data_vector_length;
        dataNodeOffsetArray[i]=dataNodeOffset;
        i++;
        cur_sample_link=cur_sample_link->next;
    }
    numSample=i;
}
