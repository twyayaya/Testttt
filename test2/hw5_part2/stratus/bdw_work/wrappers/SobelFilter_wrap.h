/****************************************************************************
*
*  Copyright (c) 2015, Cadence Design Systems. All Rights Reserved.
*
*  This file contains confidential information that may not be
*  distributed under any circumstances without the written permision
*  of Cadence Design Systems.
*
****************************************************************************/


#ifndef _SOBELFILTER_WRAP_INCLUDED_
#define _SOBELFILTER_WRAP_INCLUDED_


/* Save ioConfig define values for parent module and define those for this module's ioConfigs. */


#if defined(CYNTHVLG) 

#include <systemc.h>

#define SobelFilter_wrapper SobelFilter

/* This is the section that is seen during processing by cynthVLG of a module
 * that instantiates the module defined by this wrapper.
 */
SC_MODULE(SobelFilter)
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

	SC_CTOR(SobelFilter);
};

#elif defined(CYNTHHL) || defined(BDW_EXTRACT)

/* This is the section seen during processing by cynthHL or bdw_extract of a module 
 * that instantiates the module defined by this wrapper.
 */
#include <systemc.h>

#include "cynw_p2p.h"
#include "cyn_enums.h"


#define SobelFilter_wrapper SobelFilter

/* Only port declarations are required for nested modules.
 */
SC_MODULE(SobelFilter)
{
public:
	sc_in< bool > i_clk;
	sc_in< bool > i_rst;
	cynw::cynw_p2p_base_in <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > i_r;
	cynw::cynw_p2p_base_in <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > i_g;
	cynw::cynw_p2p_base_in <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > i_b;
	cynw::cynw_p2p_base_out <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > o_result_r;
	cynw::cynw_p2p_base_out <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > o_result_g;
	cynw::cynw_p2p_base_out <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > o_result_b;


	SC_HAS_PROCESS(SobelFilter);
    	SobelFilter_wrapper( sc_module_name name = sc_module_name( sc_gen_unique_name("SobelFilter")) );
};

#else

#include <esc.h>


/* This is the section seen during processing by gcc either when the module
 * itself is compiled, or when a module that instantiates it is compiled.
 */

struct SobelFilter;
struct SobelFilter_cosim;
struct SobelFilter_cycsim;
struct SobelFilter_rtl;

#ifdef BDW_COWARE
#include	"SobelFilter_coware.h"
#endif

#include "cynw_p2p.h"
#include "cyn_enums.h"


// Declaration of wrapper with BEH level ports

SC_MODULE(SobelFilter_wrapper)
{
public:

	sc_in< bool > i_clk;
	sc_in< bool > i_rst;
	cynw::cynw_p2p_base_in <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > i_r;
	cynw::cynw_p2p_base_in <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > i_g;
	cynw::cynw_p2p_base_in <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > i_b;
	cynw::cynw_p2p_base_out <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > o_result_r;
	cynw::cynw_p2p_base_out <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > o_result_g;
	cynw::cynw_p2p_base_out <sc_dt::sc_uint <(int)8 >, CYN::cyn_enum <(int)1 > > o_result_b;

    
    // These signals are used to connect structured ports or ports that need
    // type conversion to the RTL ports.
    
    
	enum Representation { BDWRep_None, BDWRep_Behavioral, BDWRep_RTL_C, BDWRep_RTL_HDL, BDWRep_Gates, BDWRep_CYC_HDL };
	static const char * simConfigName();
	static Representation lookupRepresentation( const char* instName );
	// create the netlist
    void InitInstances( );
    void InitThreads();

    // delete the netlist
    void DeleteInstances();

	void CloseTrace();
	void start_of_simulation();
	void end_of_simulation();


    // The following threads are used to connect structured ports to the actual
    // RTL ports.
    

	SC_HAS_PROCESS(SobelFilter_wrapper);

    	SobelFilter_wrapper( sc_module_name name = sc_module_name( sc_gen_unique_name("SobelFilter")) )
		: sc_module(name)
		  ,i_clk("i_clk")
		  	,i_rst("i_rst")
		  	,i_r("i_r")
		  	,i_g("i_g")
		  	,i_b("i_b")
		  	,o_result_r("o_result_r")
		  	,o_result_g("o_result_g")
		  	,o_result_b("o_result_b")
		  	

		  ,SobelFilter0(0), SobelFilter_cosim0(0), SobelFilter_rtl0(0), SobelFilter_cycsim0(0)
		

    {
        InitInstances( );
		InitThreads();
    }


    // destructor
    ~SobelFilter_wrapper()
    {
        DeleteInstances();

		CloseTrace();
    }

    bool isBEH() { return ( SobelFilter_wrapper::lookupRepresentation( name() ) == BDWRep_Behavioral ); }
    bool isRTL_C() { return ( SobelFilter_wrapper::lookupRepresentation( name() ) == BDWRep_RTL_C ); }
    bool isRTL_V() { return ( SobelFilter_wrapper::lookupRepresentation( name() ) == BDWRep_RTL_HDL ); }
    bool isGATES_V() { return ( SobelFilter_wrapper::lookupRepresentation( name() ) == BDWRep_Gates ); }
	bool isCosim() { return ( isRTL_V() || isGATES_V() ); }
    bool isCycsim() { return ( SobelFilter_wrapper::lookupRepresentation( name() ) == BDWRep_CYC_HDL ); }

	SobelFilter* behModule() { return SobelFilter0; }
	SobelFilter_cosim* cosimModule() { return SobelFilter_cosim0; }
	SobelFilter_cycsim* cycsimModule() { return SobelFilter_cycsim0; }
	SobelFilter_rtl* rtlModule() { return SobelFilter_rtl0; }

	SobelFilter* SobelFilter0;
	SobelFilter_cosim* SobelFilter_cosim0;
	SobelFilter_rtl* SobelFilter_rtl0;
	SobelFilter_cycsim* SobelFilter_cycsim0;
};

// Declaration of wrapper with RTL level ports

SC_MODULE(SobelFilter_wrapper_r)
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
    
    
	enum Representation { BDWRep_None, BDWRep_Behavioral, BDWRep_RTL_C, BDWRep_RTL_HDL, BDWRep_Gates, BDWRep_CYC_HDL };

	static const char * simConfigName();
	static Representation lookupRepresentation( const char* instName );
	// create the netlist
    void InitInstances();
    void InitThreads();

    // delete the netlist
    void DeleteInstances();

	void CloseTrace();
	void start_of_simulation();
	void end_of_simulation();


    // The following threads are used to connect structured ports to the actual
    // RTL ports.
    

	SC_HAS_PROCESS(SobelFilter_wrapper_r);

	SobelFilter_wrapper_r( sc_module_name name = sc_module_name( sc_gen_unique_name("SobelFilter")) )
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
		  	

		  ,SobelFilter0(0), SobelFilter_cosim0(0), SobelFilter_rtl0(0), SobelFilter_cycsim0(0)
		

    {
        InitInstances();
        InitThreads();
		end_module();
    }


    // destructor
    ~SobelFilter_wrapper_r()
    {
        DeleteInstances();

		CloseTrace();
    }

    bool isBEH() { return ( SobelFilter_wrapper_r::lookupRepresentation( name() ) == BDWRep_Behavioral ); }
    bool isRTL_C() { return ( SobelFilter_wrapper_r::lookupRepresentation( name() ) == BDWRep_RTL_C ); }
    bool isRTL_V() { return ( SobelFilter_wrapper_r::lookupRepresentation( name() ) == BDWRep_RTL_HDL ); }
    bool isGATES_V() { return ( SobelFilter_wrapper_r::lookupRepresentation( name() ) == BDWRep_Gates ); }
	bool isCosim() { return ( isRTL_V() || isGATES_V() ); }
    bool isCycsim() { return ( SobelFilter_wrapper_r::lookupRepresentation( name() ) == BDWRep_CYC_HDL ); }

	SobelFilter* behModule() { return SobelFilter0; }
	SobelFilter_cosim* cosimModule() { return SobelFilter_cosim0; }
	SobelFilter_cycsim* cycsimModule() { return SobelFilter_cycsim0; }
	SobelFilter_rtl* rtlModule() { return SobelFilter_rtl0; }

protected:
	SobelFilter* SobelFilter0;
	SobelFilter_cosim* SobelFilter_cosim0;
	SobelFilter_rtl* SobelFilter_rtl0;
	SobelFilter_cycsim* SobelFilter_cycsim0;
};

#endif

/* Restore ioConfig define values for parent module. */

#endif /*  */
