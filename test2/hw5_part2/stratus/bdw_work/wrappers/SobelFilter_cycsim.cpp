/****************************************************************************
*
*  Copyright (c) 2015, Cadence Design Systems. All Rights Reserved.
*
*  This file contains confidential information that may not be
*  distributed under any circumstances without the written permision
*  of Cadence Design Systems.
*
****************************************************************************/

#ifdef NCSC
#include <string>
typedef std::string sc_string;
#define SYSTEMC_VERSION 20070314
#endif

#include "SobelFilter_cycsim.h"
#include "esc_catrace.h"
#include "verilated.h"
#include "VSobelFilter_rtl.h"


class VSobelFilter_rtl;

class SobelFilter_cycsimV : public SobelFilter_cycsim
{
    // Port declarations inherited from parent class

    // These signals are used to connect ports that need type conversion to the RTL ports
    sc_signal< uint32_t > i_r_data_I;
    sc_signal< uint32_t > i_g_data_I;
    sc_signal< uint32_t > i_b_data_I;
    sc_signal< uint32_t > o_result_r_data_O;
    sc_signal< uint32_t > o_result_g_data_O;
    sc_signal< uint32_t > o_result_b_data_O;


    // The following threads are used to do the type conversion.
    inline void thread_i_r_data_I() {
        i_r_data_I = (sc_uint<8>)i_r_data.read();
    };
    inline void thread_i_g_data_I() {
        i_g_data_I = (sc_uint<8>)i_g_data.read();
    };
    inline void thread_i_b_data_I() {
        i_b_data_I = (sc_uint<8>)i_b_data.read();
    };
    inline void thread_o_result_r_data_O() {
        o_result_r_data = o_result_r_data_O.read();
    };
    inline void thread_o_result_g_data_O() {
        o_result_g_data = o_result_g_data_O.read();
    };
    inline void thread_o_result_b_data_O() {
        o_result_b_data = o_result_b_data_O.read();
    };


    SC_HAS_PROCESS(SobelFilter_cycsimV);

public:

    SobelFilter_cycsimV( sc_module_name in_name=sc_module_name(sc_gen_unique_name(" SobelFilter") ) )
        : SobelFilter_cycsim(in_name)
    {
        // instantiate the verilated module
        SobelFilter_cyc0 = new VSobelFilter_rtl( "VSobelFilter" );
        // generate port connections
        SobelFilter_cyc0->i_clk(i_clk);
        SobelFilter_cyc0->i_rst(i_rst);
        SobelFilter_cyc0->i_r_busy(i_r_busy);
        SobelFilter_cyc0->i_r_vld(i_r_vld);
        SobelFilter_cyc0->i_r_data(i_r_data_I);
        SobelFilter_cyc0->i_g_busy(i_g_busy);
        SobelFilter_cyc0->i_g_vld(i_g_vld);
        SobelFilter_cyc0->i_g_data(i_g_data_I);
        SobelFilter_cyc0->i_b_busy(i_b_busy);
        SobelFilter_cyc0->i_b_vld(i_b_vld);
        SobelFilter_cyc0->i_b_data(i_b_data_I);
        SobelFilter_cyc0->o_result_r_busy(o_result_r_busy);
        SobelFilter_cyc0->o_result_r_vld(o_result_r_vld);
        SobelFilter_cyc0->o_result_r_data(o_result_r_data_O);
        SobelFilter_cyc0->o_result_g_busy(o_result_g_busy);
        SobelFilter_cyc0->o_result_g_vld(o_result_g_vld);
        SobelFilter_cyc0->o_result_g_data(o_result_g_data_O);
        SobelFilter_cyc0->o_result_b_busy(o_result_b_busy);
        SobelFilter_cyc0->o_result_b_vld(o_result_b_vld);
        SobelFilter_cyc0->o_result_b_data(o_result_b_data_O);


        // setup the type conversion threads
        SC_METHOD( thread_i_r_data_I);
            sensitive << i_r_data;
        SC_METHOD( thread_i_g_data_I);
            sensitive << i_g_data;
        SC_METHOD( thread_i_b_data_I);
            sensitive << i_b_data;
        SC_METHOD( thread_o_result_r_data_O);
            sensitive << o_result_r_data_O;
        SC_METHOD( thread_o_result_g_data_O);
            sensitive << o_result_g_data_O;
        SC_METHOD( thread_o_result_b_data_O);
            sensitive << o_result_b_data_O;


        TraceFileType* tfp = esc_get_ca_trace_file();
        if ( tfp )
        {
            SobelFilter_cyc0->trace(tfp,99);
        }
    };

    ~SobelFilter_cycsimV()
    {
        delete SobelFilter_cyc0;
    }

    void end_of_elaboration()
    {
        esc_open_ca_trace_file();
    }

    void end_of_simulation()
    {
        esc_close_ca_trace_file();
    }

protected:
    VSobelFilter_rtl* SobelFilter_cyc0;
};

SobelFilter_cycsim* SobelFilter_cycsim_factory()
{
    return new SobelFilter_cycsimV("SobelFilter");
}

class SobelFilter_linkup
{
public:
    SobelFilter_linkup() {
        extern SobelFilter_cycsim* (*SobelFilter_cycsim_factory_p)();
        SobelFilter_cycsim_factory_p = &SobelFilter_cycsim_factory;
    }
};

// Create the one instance of the factory for this verilated model.
SobelFilter_linkup SobelFilter_cycsimV_link;


