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

    int resultArray[SHARD_SIZE];
    for(int i=0;i<SHARD_SIZE;i++) resultArray[i]=1;

    //ofstream fout("out.txt");
    ofstream fout;
    fout.open("out.txt");
    ofstream ftimeout("time.txt");
    int numSampleArray[2];
    for(int i=0;i<2;i++) numSampleArray[i]=0;
    clock_t start;
    start=clock();

    while(1){
        M3::load_test_data(shardz);

        while(1){
            clock_t start,end;

            cout<<endl<<"shardx,shardy: "<<shardx<<' '<<shardy<<endl;
            ftimeout<<"shardx,shardy: "<<shardx<<' '<<shardy<<endl;

            start=clock();

            if (M3::flag_train()){

                M3::load_train_data(shardx,shardy);
                //M3::divide_train_data();
                //M3::training_train_data();
            }

            ftimeout<<"load time: "<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;

            start=clock();

            if (M3::flag_classify()){
                M3::classify_test_data(resultArray);
            }

            ftimeout<<"classify time: "<<(clock()-start)/(double) CLOCKS_PER_SEC<<endl;

            cout<<"offset: "<<M3::getFileOffset(0)<<' '<<M3::getFileOffset(1)<<endl;

            if(shardy==0) numSampleArray[0]+=M3::getSampleNum(0);
            if(shardx==0) numSampleArray[1]+=M3::getSampleNum(1);

            if(M3::getFileOffset(1)==-1) 
                for(int i=0;i<M3::getSampleNum(0);i++)
                    fout<<resultArray[i]<<endl;
            if(M3::getFileOffset(0)==-1 && M3::getFileOffset(1)==-1) break;
            shardy++;
            if(M3::getFileOffset(1)==-1) {
                for(int i=0;i<SHARD_SIZE;i++) {
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
