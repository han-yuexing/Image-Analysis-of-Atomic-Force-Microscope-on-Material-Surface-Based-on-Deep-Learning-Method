/*
 *  @(#)$Id: file.h $
 *  @(#) Declaration file of class TST_ProductionYearFunction
 *
 *  Copyright @ 2013 Suntec Software(Shanghai) Co., Ltd.
 *  All Rights Reserved.
 */
/*  @file   
 *  Declaration of TST_ProductionYearFunction
 * 
 *  TO DESC : FILE DETAIL DESCRIPTION, BUT DON'T DESCRIBE CLASS DETAIL HERE.
 */
#ifndef _FILE_H_
#define _FILE_H_
#include <string>
#include <vector>
using namespace std;
// Class declaration
class file;
/*-----------------------------------------------------------------------*/
/**
 *  class file
 *
 * TO DESC : CLASS DETAIL DESCRIPTION
 * 
 * @author $Author$
 * @version $Revision$
 * @date     $Date::                      #$
 */
/*-----------------------------------------------------------------------*/
class file
{
public:
	/*-----------------------------------------------------------------------*/
	/**
	 * getJustCurrentFile
	 * 
	 * TO DESC : FUNCTION DESCRIPTION
	 * 
	 * @param string path:file path
	 *		  vector<string>& files:vector for store file name
	 * 
	 * @return void
	 */
	/*-----------------------------------------------------------------------*/
	static void getJustCurrentFile( string path, vector<string>& files);
};
#endif //_FILE_H_
/* EOF */
