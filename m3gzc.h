#ifndef _M3GZC_H 
#define _M3GZC_H

#include<iostream>
#include<cmath>
#include"SerializedSampleSet.h"

int*  m3gzcGPU(SerializedSampleSet sss1,SerializedSampleSet sss2,SerializedSampleSet sss3);

int*  m3gzcCPU(SerializedSampleSet sss1,SerializedSampleSet sss2,SerializedSampleSet sss3);

#endif

