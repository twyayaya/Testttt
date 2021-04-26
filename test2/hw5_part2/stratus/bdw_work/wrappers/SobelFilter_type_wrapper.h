/****************************************************************************
*
*  Copyright (c) 2015, Cadence Design Systems. All Rights Reserved.
*
*  This file contains confidential information that may not be
*  distributed under any circumstances without the written permision
*  of Cadence Design Systems.
*
****************************************************************************/
/****************************************************************************
*
* This file contains the SobelFilter_type_wrapper module
* for use in the verilog verification wrapper SobelFilter_vlwrapper.v
* It creats an instance of the SobelFilter module and has threads
* for converting the BEH ports to RTL level ports on the wrapper.
*
****************************************************************************/


#ifndef _SOBELFILTER_TYPE_WRAP_INCLUDED_
#define _SOBELFILTER_TYPE_WRAP_INCLUDED_


#include <systemc.h>
#include "SobelFilter.h"


// Declaration of wrapper with RTL level ports

SC_MODULE(SobelFilter_type_wrapper)
{
public:

	sc_in< bool > i_clk;
	sc_in< bool > i_rst;
	sc_out< bool > i_r_busy;
	sc_in< bool > i_r_vld;
	sc_in< sc_uint< 8 > > i_r_data;
	sc_out< bool > i_g_busy;
	sc_in< bool > i_g_vld;
	sc_in< sc_uint< 8 > > i_g_data;
	sc_out< bool > i_b_busy;
	sc_in< bool > i_b_vld;
	sc_in< sc_uint< 8 > > i_b_data;
	sc_in< bool > o_result_r_busy;
	sc_out< bool > o_result_r_vld;
	sc_out< sc_uint< 8 > > o_result_r_data;
	sc_in< bool > o_result_g_busy;
	sc_out< bool > o_result_g_vld;
	sc_out< sc_uint< 8 > > o_result_g_data;
	sc_in< bool > o_result_b_busy;
	sc_out< bool > o_result_b_vld;
	sc_out< sc_uint< 8 > > o_result_b_data;

    
    // These signals are used to connect structured ports or ports that need
    // type conversion to the RTL ports.
    
    

	// create the netlist
    void InitInstances();
    void InitThreads();

    // delete the netlist
    void DeleteInstances();

    // The following threads are used to connect structured ports to the actual
    // RTL ports.
    

	SC_HAS_PROCESS(SobelFilter_type_wrapper);

	SobelFilter_type_wrapper( sc_module_name name = sc_module_name( sc_gen_unique_name("SobelFilter")) )
		: sc_module(name)
		  ,i_clk("i_clk")
		  	,i_rst("i_rst")
		  	,i_r_busy("i_r_busy")
		  	,i_r_vld("i_r_vld")
		  	,i_r_data("i_r_data")
		  	,i_g_busy("i_g_busy")
		  	,i_g_vld("i_g_vld")
		  	,i_g_data("i_g_data")
		  	,i_b_busy("i_b_busy")
		  	,i_b_vld("i_b_vld")
		  	,i_b_data("i_b_data")
		  	,o_result_r_busy("o_result_r_busy")
		  	,o_result_r_vld("o_result_r_vld")
		  	,o_result_r_data("o_result_r_data")
		  	,o_result_g_busy("o_result_g_busy")
		  	,o_result_g_vld("o_result_g_vld")
		  	,o_result_g_data("o_result_g_data")
		  	,o_result_b_busy("o_result_b_busy")
		  	,o_result_b_vld("o_result_b_vld")
		  	,o_result_b_data("o_result_b_data")
		  	

		  ,SobelFilter0(0)
		

    {
        InitInstances();
        InitThreads();
		end_module();
    }


    // destructor
    ~SobelFilter_type_wrapper()
    {
        DeleteInstances();
    }


protected:
	SobelFilter* SobelFilter0;
};

#endif /*  */
