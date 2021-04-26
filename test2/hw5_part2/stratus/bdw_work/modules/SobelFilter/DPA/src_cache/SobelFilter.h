#ifndef SOBEL_FILTER_H_
#define SOBEL_FILTER_H_
#include <systemc>
using namespace sc_core;
using namespace sc_dt;
using namespace std;

#ifndef NATIVE_SYSTEMC
#include <cynw_p2p.h>
#endif

#include <sys/time.h>
#include "filter_def.h"

class SobelFilter : public sc_module
{
public:
	sc_in_clk i_clk;
	sc_in<bool> i_rst;
#ifndef NATIVE_SYSTEMC
	//cynw_p2p<sc_dt::sc_uint<24> >::in i_rgb;
	cynw_p2p<sc_dt::sc_uint<8> >::in i_r;
	cynw_p2p<sc_dt::sc_uint<8> >::in i_g;
	cynw_p2p<sc_dt::sc_uint<8> >::in i_b;

	//cynw_p2p<sc_dt::sc_uint<24> >::out o_result;
	cynw_p2p<sc_dt::sc_uint<8> >::out o_result_r;
	cynw_p2p<sc_dt::sc_uint<8> >::out o_result_g;
	cynw_p2p<sc_dt::sc_uint<8> >::out o_result_b;
#else
	//sc_fifo_in<sc_dt::sc_uint<24> > i_rgb;
	sc_fifo_in<sc_dt::sc_uint<8> > i_r;
	sc_fifo_in<sc_dt::sc_uint<8> > i_g;
	sc_fifo_in<sc_dt::sc_uint<8> > i_b;

	//sc_fifo_out<sc_dt::sc_uint<24> > o_result;
	sc_fifo_out<sc_dt::sc_uint<8> > o_result_r;
	sc_fifo_out<sc_dt::sc_uint<8> > o_result_g;
	sc_fifo_out<sc_dt::sc_uint<8> > o_result_b;
#endif

	SC_HAS_PROCESS(SobelFilter);
	SobelFilter(sc_module_name n);
	~SobelFilter();

private:
	void do_filter();
};
#endif
