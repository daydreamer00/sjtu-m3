#include "M3.h"
#include "util.h"
//using namespace M3;
int main(int argc,char ** argv){

    M3::parse();			// parse m3.config

    //debug
    cout<<"parse config file complete"<<endl;

    M3::initialize(argc,argv);	// initialize

    int shardx=0,shardy=0;

    int resultArray[SHARD_SIZE];
    for(int i=0;i<SHARD_SIZE;i++) resultArray[i]=1;

    while(M3::getFileOffset(0)!=-1 && 
            M3::getFileOffset(1)!=-1){

        if(M3::getFileOffset(1)==-1) {
            for(int i=0;i<SHARD_SIZE;i++) {
                resultArray[i]=1;
                M3::setEnableFlag(i,true);
            }
            shardx++;
            shardy=0;
            M3::setFileOffset(1,0);
        }

        if (M3::flag_train()){

            M3::load_train_data(shardx,shardy);
            //M3::divide_train_data();
            //M3::training_train_data();
        }

        if (M3::flag_classify()){
            M3::classify_test_data(resultArray);
        }

        shardy++;
    }

    //if (M3::flag_score()){
    //    M3::score_test_data();
    //}

    //if (M3::flag_compare()){
    //    M3::compare_true_label();
    //}
    M3::finalize();

    return 0;
}
