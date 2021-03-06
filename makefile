##################################################################
# No warranty, no liability
# You use this at your own risk
#################################################################
#OBJS=m3_main.o M3.o svm.o libsvm.o libsvm_parameter.o m3_factory.o prior_divide.o hyper_plane.o centroid.o random_divide.o divider.o hc_voter.o m3_parameter.o data_split.o svm_hideo.o svm_common.o svm_learn.o svm_light_parameter.o svm_light.o util.o SerializedSampleSet.o  m3gzc.o #cuPrintf.o #m3gzcKernel.o
OBJS=m3_main.o M3.o m3_parameter.o data_split.o util.o SerializedSampleSet.o  m3gzc.o #cuPrintf.o #m3gzcKernel.o
CPP=g++#mpic++
CC=gcc#mpicc
CFLAGS=-Wall -g
#NVCCFLAGS=-g -O -Xcompiler -rdynamic -lineinfo -arch=sm_20 #debug version
NVCCFLAGS= -O -arch=sm_20 
#CFLAGS=-pg

m3train: $(OBJS)
	$(CPP) -L/opt/cuda/lib64 -lcudart $(OBJS) -o m3 

m3_main.o: m3_main.cpp M3.h
	$(CPP) $(CFLAGS) -c m3_main.cpp -o m3_main.o

M3.o: M3.cpp M3.h 
	$(CPP) $(CFLAGS) -c M3.cpp -o M3.o

util.o: util.cpp util.h
	$(CPP) $(CFLAGS) -c util.cpp -o util.o

SerializedSampleSet.o: SerializedSampleSet.cpp SerializedSampleSet.h  
	$(CPP) $(CFLAGS) -c SerializedSampleSet.cpp -o SerializedSampleSet.o

m3gzc.o: m3gzc.h m3gzc.cu 
	nvcc $(NVCCFLAGS)  -c m3gzc.cu -o m3gzc.o

#cuPrintf.o: cuPrintf.cuh cuPrintf.cu
#	nvcc -c cuPrintf.cu -o cuPrintf.o

#m3gzcKernel.o: m3gzc_kernel.cu m3gzc.h m3gzc.cpp
#	nvcc -c m3gzc_kernel.cu -o m3gzcKernel.o

svm.o: svm.cpp svm.h
	$(CPP) $(CFLAGS) -c svm.cpp -o svm.o

libsvm.o: libsvm.cpp libsvm.h classifier.h 
	$(CPP) $(CFLAGS) -c libsvm.cpp -o libsvm.o

libsvm_parameter.o: libsvm_parameter.cpp libsvm_parameter.h
	$(CPP) $(CFLAGS) -c libsvm_parameter.cpp -o libsvm_parameter.o

svm_learn.o: svm_learn.c svm_learn.h
	$(CC) -c $(CFLAGS) svm_learn.c -o svm_learn.o

svm_hideo.o: svm_hideo.c
	$(CC) -c $(CFLAGS) svm_hideo.c -o svm_hideo.o

svm_common.o: svm_common.h svm_common.c kernel.h
	$(CC) -c $(CFLAGS) svm_common.c -o svm_common.o

svm_light_parameter.o:svm_light_parameter.cpp svm_light_parameter.h 
	$(CPP) $(CFLAGS) -c svm_light_parameter.cpp -o svm_light_parameter.o

svm_light.o: svm_light.cpp svm_light.h classifier.h
	$(CPP) $(CFLAGS) -c svm_light.cpp -o svm_light.o

m3_factory.o: m3_factory.cpp m3_factory.h
	$(CPP) $(CFLAGS) -c m3_factory.cpp -o m3_factory.o

hyper_plane.o: hyper_plane.cpp hyper_plane.h
	$(CPP) $(CFLAGS) -c hyper_plane.cpp -o hyper_plane.o
centroid.o: centroid.cpp centroid.h
	$(CPP) $(CFLAGS) -c centroid.cpp -o centroid.o

random_divide.o: random_divide.cpp random_divide.h
	$(CPP) $(CFLAGS) -c random_divide.cpp -o random_divide.o

prior_divide.o: prior_divide.cpp prior_divide.h
	$(CPP) $(CFLAGS) -c prior_divide.cpp -o prior_divide.o
	

divider.o: divider.cpp divider.h
	$(CPP) $(CFLAGS) -c divider.cpp -o divider.o

hc_voter.o: hc_voter.cpp hc_voter.h voter.h
	$(CPP) $(CFLAGS) -c hc_voter.cpp -o hc_voter.o

m3_parameter.o: m3_parameter.cpp parameter.h
	$(CPP) $(CFLAGS) -c m3_parameter.cpp -o m3_parameter.o

data_split.o: data_split.cpp data_split.h
	$(CPP) $(CFLAGS) -c data_split.cpp -o data_split.o

clean: 
	rm -f *.o m3

