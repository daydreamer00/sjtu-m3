#ifndef _UTIL_H
#define _UTIL_H
#include<cstdlib>
#include<string>
#include<iostream>
#include<ctime>

#define SHARD_SIZE 32 
#define TEST_SHARD_SIZE 128 
#define BLOCK_SIZE 8 

#define TIMER_BEGIN(t) t=clock()
#define TIMER_PRINT(msg,t) cout<<msg<<" time: "<<(clock()-t)/(double)CLOCKS_PER_SEC<<endl

#define AVERAGE_DATA_PER_SAMPLE 200
#define SQUARE(x) (x*x)
#define THRESHOLD 0.6


using namespace std;

struct Data_Node{
  int index;
  float value;
};

struct Data_Sample{
  int index;
  float label;
  int data_vector_length;
  Data_Node * data_vector;
};

struct Sample_Link{
    Data_Sample * sample_head;
    Sample_Link * next;
    int length;
};

struct Divide_Info{
  int start_offset,end_offset;
  int length;
};

struct Subset_Info{
  float label_1,label_2;
  int subset_num_1,subset_num_2;
  int save_index;
  int subset_memory;
  int process_rank;
  int process_1,process_2;
  int start_1,start_2;
  int end_1,end_2;

  const bool operator < (const Subset_Info & si) const {
    if (label_1<si.label_1)
      return true;
    else if (label_1>si.label_1)
      return false;

    if(label_2<si.label_2)
      return true;
    else if (label_2>si.label_2)
      return false;

    if (process_1<si.process_1)
      return true;
    else if (process_1>si.process_1)
      return false;

    if (process_2<si.process_2)
      return true;
    else if (process_2>si.process_2)
      return false;

    if (start_1<si.start_1)
      return true;
    else if (start_1>si.start_1)
      return false;

    if (start_2<si.start_2)
      return true;
    else if (start_2>si.start_2)
      return false;

    return false;
  }
};

void error(string msg);

void assert_true(bool flag);
#endif
