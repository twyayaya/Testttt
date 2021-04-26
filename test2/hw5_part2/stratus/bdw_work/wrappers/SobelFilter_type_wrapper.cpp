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

#include "SobelFilter_type_wrapper.h"


SC_MODULE_EXPORT(SobelFilter_type_wrapper)

// The following threads are used to connect RTL ports to the actual
// structured ports

void SobelFilter_type_wrapper::InitInstances()
{
    SobelFilter0 = new SobelFilter( "SobelFilter" );

    SobelFilter0->i_clk(i_clk);
    SobelFilter0->i_rst(i_rst);
    SobelFilter0->i_r.busy(i_r_busy);
    SobelFilter0->i_r.vld(i_r_vld);
    SobelFilter0->i_r.data(i_r_data);
    SobelFilter0->i_g.busy(i_g_busy);
    SobelFilter0->i_g.vld(i_g_vld);
    SobelFilter0->i_g.data(i_g_data);
    SobelFilter0->i_b.busy(i_b_busy);
    SobelFilter0->i_b.vld(i_b_vld);
    SobelFilter0->i_b.data(i_b_data);
    SobelFilter0->o_result_r.busy(o_result_r_busy);
    SobelFilter0->o_result_r.vld(o_result_r_vld);
    SobelFilter0->o_result_r.data(o_result_r_data);
    SobelFilter0->o_result_g.busy(o_result_g_busy);
    SobelFilter0->o_result_g.vld(o_result_g_vld);
    SobelFilter0->o_result_g.data(o_result_g_data);
    SobelFilter0->o_result_b.busy(o_result_b_busy);
    SobelFilter0->o_result_b.vld(o_result_b_vld);
    SobelFilter0->o_result_b.data(o_result_b_data);

}

void SobelFilter_type_wrapper::InitThreads()
{
    
}

void SobelFilter_type_wrapper::DeleteInstances()
{
        delete SobelFilter0;
        SobelFilter0 = 0;
}

