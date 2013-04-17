#include "util.h"
#include<iostream>

using namespace std;

void error(string msg){
    cout<<"Error: "<<msg<<endl;
    exit(1);
}

void assert_true(bool flag){
    if(!flag) error("Should be true");
}
