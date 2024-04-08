/*
 *  @(#)$Id: file.cpp $
 *  @(#) Implementation file of class file
 *
 *  Copyright @ 2015 Shanghai University
 *  All Rights Reserved.
 */
/*  @file   
 *  Implementation of file
 * 
 *  TO DESC : FILE DETAIL DESCRIPTION, BUT DON'T DESCRIBE CLASS DETAIL HERE.
 */
//#ifdef _WIN32
//#include<io.h>
//#else
#include<unistd.h>
#include<stdio.h>
#include<dirent.h>
#include<sys/stat.h>
//#endif
#include<fstream>
#include<string>
#include<string.h>
#include<vector>
#include"file.h"
/*************************************************************************

    getJustCurrentFile

    [Description]

        TO DESC

*************************************************************************/
void file::getJustCurrentFile( string path, vector<string>& files)
{  
//#ifndef WIN32
	//文件句柄
	//long  hFile  =  0;
	//文件信息
	//struct _finddata_t fileinfo;
	//string p;
	//if ((hFile = _findfirst(p.assign(path).append("/*").c_str(),&fileinfo)) != -1)
	//{  
		//do 
		//{   
			//if ((fileinfo.attrib & _A_SUBDIR)) 
			//{  
				//; 
			//} 
			//else
			//{
				//files.push_back(fileinfo.name);
				////files.push_back(p.assign(path).append("\\").append(fileinfo.name) );
			//}
		//}while(_findnext(hFile, &fileinfo) == 0);
		//_findclose(hFile);
	//}
//#else
	DIR *dp;
	struct dirent* entry;
	struct stat statbuf;
	if ((dp = opendir(path.c_str())) == NULL) {
		fprintf(stderr, "cannot open directory: %s\n", path.c_str());
		return;
	}
	chdir(path.c_str());
	while ((entry = readdir(dp)) != NULL) {
		lstat(entry->d_name, &statbuf);
		if (S_ISDIR(statbuf.st_mode)) {
			if (strcmp(".", entry->d_name) == 0 || strcmp("..", entry->d_name) == 0) {
				continue;
			}
		} else {
			string filename = entry->d_name;
			files.push_back(filename);
		}
	}
	chdir("..");
	closedir(dp);
//#endif
}
/* EOF */
