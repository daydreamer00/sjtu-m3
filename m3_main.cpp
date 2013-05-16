#include "M3.h"
#include "util.h"
#include<fstream>
#include<ctime>
#include<iostream>

using namespace std;
//using namespace M3;
int main(int argc,char ** argv){

    M3::parse();			// parse m3.config

    //debug
    cout<<"parse config file complete"<<endl;

    M3::initialize(argc,argv);	// initialize

    int shardx=0,shardy=0,shardz=0;

    int resultArray[SHARD_SIZE*TEST_SHARD_SIZE];
    for(int i=0;i<SHARD_SIZE*TEST_SHARD_SIZE;i++) resultArray[i]=1;

    //ofstream fout("out.txt");
    char outDirNameArray[100];
    sprintf(outDirNameArray,"data_%d_%d_%d",SHARD_SIZE,TEST_SHARD_SIZE,BLOCK_SIZE);
    string outDirNameString(outDirNameArray);
    system(string("mkdir "+outDirNameString).c_str());
    outDirNameString+="/";
    ofstream fout;
    fout.open(string(outDirNameString+"out.txt").c_str());
    ofstream ftimeout(string(outDirNameString+"time.txt").c_str());
    int numSampleArray[3];
    for(int i=0;i<3;i++) numSampleArray[i]=0;
    clock_t start;
    start=clock();

    while(1){
        M3::load_test_data(shardz);
        numSampleArray[2]+=M3::getSampleNum(2);

        while(1){
            clock_t start,end;

            cout<<endl<<"shardx,shardy,shardz: "<<shardx<<' '<<shardy<<' '<<shardz<<endl;
            //ftimeout<<"shardx,shardy: "<<shardx<<' '<<shardy<<endl;

            start=clock();

            if (M3::flag_train()){

                M3::load_train_data(shardx,shardy);
                //M3::divide_train_data();
                //M3::training_train_data();
            }

            //ftimeout<<"load time: "<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;
            ftimeout<<(clock()-start)/(double) CLOCKS_PER_SEC<<' ';

            start=clock();

            if (M3::flag_classify()){
                M3::classify_test_data(resultArray);
            }

            //ftimeout<<"classify time: "<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;
            ftimeout<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;

            cout<<"offset: "<<M3::getFileOffset(0)<<' '<<M3::getFileOffset(1)<<endl;

            //for(int i=0;i<M3::getSampleNum(0)*M3::getSampleNum(2);i++)
            //    cout<<i/M3::getSampleNum(0)<<' '<<i%M3::getSampleNum(0)<<' '<<resultArray[i]<<endl;

            if(shardy==0) numSampleArray[0]+=M3::getSampleNum(0);
            if(shardx==0) numSampleArray[1]+=M3::getSampleNum(1);

            if(M3::getFileOffset(1)==-1) 
                for(int i=0;i<M3::getSampleNum(0)*M3::getSampleNum(2);i++)
                    fout<<i/M3::getSampleNum(0)<<' '<<i%M3::getSampleNum(0)<<' '<<resultArray[i]<<endl;
            if(M3::getFileOffset(0)==-1 && M3::getFileOffset(1)==-1) {
                for(int i=0;i<TEST_SHARD_SIZE*SHARD_SIZE;i++) {
                    resultArray[i]=1;
                    M3::setEnableFlag(i,true);
                }
                shardx=shardy=0;
                M3::setFileOffset(0,0);
                M3::setFileOffset(1,0);
                break;
            }
            shardy++;
            if(M3::getFileOffset(1)==-1) {
                for(int i=0;i<TEST_SHARD_SIZE*SHARD_SIZE;i++) {
                    resultArray[i]=1;
                    M3::setEnableFlag(i,true);
                }
                shardx++;
                shardy=0;
                M3::setFileOffset(1,0);
            }


        }
        if (M3::getFileOffset(2)==-1) break;
        shardz++;
    }

    ftimeout<<"total time: "<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;
    cout<<"sample no: "<<numSampleArray[0]<<' '<<numSampleArray[1]<<endl;

    //if (M3::flag_score()){
    //    M3::score_test_data();
    //}

    //if (M3::flag_compare()){
    //    M3::compare_true_label();
    //}
    fout.close();
    ftimeout.close();
    M3::finalize();

    return 0;
}
