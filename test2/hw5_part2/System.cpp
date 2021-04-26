#include "System.h"
System::System(sc_module_name n) : sc_module(n),
								   tb("tb"), sobel_filter("sobel_filter"), clk("clk", CLOCK_PERIOD, SC_NS), rst("rst")
{
	tb.i_clk(clk);
	tb.o_rst(rst);
	sobel_filter.i_clk(clk);
	sobel_filter.i_rst(rst);
	//tb.o_rgb(rgb);
	tb.o_r(r);
	tb.o_g(g);
	tb.o_b(b);

	//tb.i_result(result);
	tb.i_result_r(result_r);
	tb.i_result_g(result_g);
	tb.i_result_b(result_b);

	//sobel_filter.i_rgb(rgb);
	sobel_filter.i_r(r);
	sobel_filter.i_g(g);
	sobel_filter.i_b(b);

	//sobel_filter.o_result(result);
	sobel_filter.o_result_r(result_r);
	sobel_filter.o_result_g(result_g);
	sobel_filter.o_result_b(result_b);

	tb.read_bmp("lena.bmp");
}

System::~System()
{
	tb.write_bmp("lena_hw5_part2_test1.bmp");
	cout << "stop" << endl;
}
