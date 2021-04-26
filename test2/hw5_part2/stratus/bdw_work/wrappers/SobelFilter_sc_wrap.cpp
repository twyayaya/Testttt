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
* This file is used to wrap the three different versions of the DUT
* block called "SobelFilter". By default, it will include the behavioral
* model. Otherwise, it will include the RTL C++ or the RTL Verilog
* depending on the definition of either of "RTL" or "COSIM".
*
****************************************************************************/


#include	"SobelFilter_sc_wrap.h"
#include	"SobelFilter_sc_foreign.h"

// The following threads are used to connect structured ports to the actual
// RTL ports

void SobelFilter_wrapper::InitInstances(  )
{
	
            
    SobelFilter0 = new SobelFilter( "SobelFilter" );

    SobelFilter0->i_clk(i_clk);
    SobelFilter0->i_rst(i_rst);
    SobelFilter0->i_r_busy(i_r.busy);
    SobelFilter0->i_r_vld(i_r.vld);
    SobelFilter0->i_r_data(i_r.data);
    SobelFilter0->i_g_busy(i_g.busy);
    SobelFilter0->i_g_vld(i_g.vld);
    SobelFilter0->i_g_data(i_g.data);
    SobelFilter0->i_b_busy(i_b.busy);
    SobelFilter0->i_b_vld(i_b.vld);
    SobelFilter0->i_b_data(i_b.data);
    SobelFilter0->o_result_r_busy(o_result_r.busy);
    SobelFilter0->o_result_r_vld(o_result_r.vld);
    SobelFilter0->o_result_r_data(o_result_r.data);
    SobelFilter0->o_result_g_busy(o_result_g.busy);
    SobelFilter0->o_result_g_vld(o_result_g.vld);
    SobelFilter0->o_result_g_data(o_result_g.data);
    SobelFilter0->o_result_b_busy(o_result_b.busy);
    SobelFilter0->o_result_b_vld(o_result_b.vld);
    SobelFilter0->o_result_b_data(o_result_b.data);

}

void SobelFilter_wrapper::InitThreads()
{
    
}

void SobelFilter_wrapper::DeleteInstances()
{
    if (SobelFilter0)
    {
        delete SobelFilter0;
        SobelFilter0 = 0;
    }
}

