#ifndef _M3_PARAMETER_CPP
#define _M3_PARAMETER_CPP

#include "parameter.h"
#include <fstream>
#include <iostream>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <cstring>
using namespace std;

M3_Parameter::M3_Parameter(){
	init("m3.config");
}

M3_Parameter::M3_Parameter(string pfile_name){
	init(pfile_name);
}

void M3_Parameter::init(string pfile_name){

	flag_train=false;
	flag_classify=false;
	flag_compare=false;
	flag_score=false;
	m3_min_max_mode=0;
	m3_load_mode=0;
	m3_classify_mode=0;
	m3_pruning_mode=0;
	m3_pruning_speed_mode=0;
	m3_pruning_combine_score=0;
    m3_version=0;

	ifstream config(pfile_name.c_str());
	string tmp;
	while (getline(config,tmp)){
		bool flag=true;

		string ll,rr;
		ll="";
		rr="";

		for (int i=0;i<tmp.length();i++){
			if (tmp[i]=='#')
				break;
			if (tmp[i]==' ')
				continue;
			if (tmp[i]!='='){
				if (flag)
					ll+=tmp[i];
				else rr+=tmp[i];
			}
			else flag=false;
		}

		if (ll=="running_process_num")
			running_process_num=atoi(rr.c_str());
		if (ll=="subset_size")
			subset_size=atoi(rr.c_str());
		if (ll=="modular_size")
			modular_size=atoi(rr.c_str());
		if (ll=="divider_rank")
			divider_rank=atoi(rr.c_str());
		if (ll=="classifier_rank")
			classifier_rank=atoi(rr.c_str());
		if (ll=="classifier_parameter_rank")
			classifier_parameter_rank=atoi(rr.c_str());
		if (ll=="voter_rank")
			voter_rank=atoi(rr.c_str());
		if (ll=="train"){
			flag_train=true;
			train_data=rr;
		}
		if (ll=="classify"){
			flag_classify=true;
			classify_data=rr;
		}
		if (ll=="compare"){
			flag_compare=true;
			compare_label_data=rr;
		}
		if (ll=="score"){
			flag_score=true;
			classify_data=rr;
		}
		if (ll=="m3_min_max_mode")
			m3_min_max_mode=atoi(rr.c_str());
		if (ll=="m3_load_mode")
			m3_load_mode=atoi(rr.c_str());
		if (ll=="m3_classify_mode")
			m3_classify_mode=atoi(rr.c_str());
		if (ll=="m3_pruning_mode")
			m3_pruning_mode=atoi(rr.c_str());
		if (ll=="m3_pruning_speed_mode")
			m3_pruning_speed_mode=atoi(rr.c_str());
		if (ll=="m3_pruning_combine_score"){
			m3_pruning_speed_mode=1;
			m3_pruning_combine_score=atoi(rr.c_str());
		}
		if (ll=="version") 
            m3_version=atoi(rr.c_str());
	}

	config.close();
}

void M3_Parameter::parse_as_cmd(string pfile_name,int ** argc,char *** argv){
	vector<string> cmd;
	ifstream cmd_file(pfile_name.c_str());
	cmd.clear();
	cmd.push_back("m3");

	string tmp;
	while (cmd_file >> tmp){
		cmd.push_back(tmp);
	}
	cmd_file.close();

	*argv=new char* [cmd.size()];
	*argc=new int(cmd.size());
	
	for (int i=0;i<(**argc);i++){
		int len=cmd[i].length();
		(*argv)[i]=new char[len];
		strcpy((*argv)[i],cmd[i].c_str());
	}
}

void M3_Parameter::rm_parse_as_cmd(int ** argc,char *** argv){

	for (int i=0;i<(**argc);i++)
		delete [] (*argv)[i];

	delete [] (*argv);
	delete (*argc);
}


#endif
