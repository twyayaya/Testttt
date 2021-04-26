//
// Verilog co-simulation wrapper module for the SobelFilter module.

// This module contains the followng items:
//	- An instance of the verilog RTL module created by cynthVLG.
//	- Wires and reg's connected to the ports of the RTL module.
//	- A clock generator using the project's clock frequency and connected to
//    the RTL module's clock port.
//
// The wires and reg's will be connected to sc_signals in the SystemC simulaton
// by esc_link_signals calls in the proxy module found in the SobelFilter_cosim.h 
// file.

`timescale 1 ps / 1 ps

module SobelFilter_cosim;

	reg i_clk;
	reg i_rst;
	wire i_r_busy;
	reg i_r_vld;
	reg[7:0] i_r_data;
	wire i_g_busy;
	reg i_g_vld;
	reg[7:0] i_g_data;
	wire i_b_busy;
	reg i_b_vld;
	reg[7:0] i_b_data;
	reg o_result_r_busy;
	wire o_result_r_vld;
	wire[7:0] o_result_r_data;
	reg o_result_g_busy;
	wire o_result_g_vld;
	wire[7:0] o_result_g_data;
	reg o_result_b_busy;
	wire o_result_b_vld;
	wire[7:0] o_result_b_data;


	`include "hub.v"

	SobelFilter SobelFilter0( .i_clk(i_clk),
	 .i_rst(i_rst),
	 .i_r_busy(i_r_busy),
	 .i_r_vld(i_r_vld),
	 .i_r_data(i_r_data),
	 .i_g_busy(i_g_busy),
	 .i_g_vld(i_g_vld),
	 .i_g_data(i_g_data),
	 .i_b_busy(i_b_busy),
	 .i_b_vld(i_b_vld),
	 .i_b_data(i_b_data),
	 .o_result_r_busy(o_result_r_busy),
	 .o_result_r_vld(o_result_r_vld),
	 .o_result_r_data(o_result_r_data),
	 .o_result_g_busy(o_result_g_busy),
	 .o_result_g_vld(o_result_g_vld),
	 .o_result_g_data(o_result_g_data),
	 .o_result_b_busy(o_result_b_busy),
	 .o_result_b_vld(o_result_b_vld),
	 .o_result_b_data(o_result_b_data) );


	// Variables that can be set externally
	reg i_clk_firstEdge;
	reg i_clk_initialized;
	reg i_clk_started;
	real i_clk_startTime;
	real i_clk_firstHalf;
	real i_clk_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    i_clk_firstEdge = 1'b1;
	    i_clk_startTime = 0.000;		// Offset from time 0 to first edge.
	    i_clk_firstHalf = 5.000;	// Delay for first half of cycle.
	    i_clk_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on i_clk_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( i_clk_initialized !== 1'b1 )
	        @( i_clk_initialized );
	
	    if ( i_clk_startTime > 0.0 ) begin
	        i_clk_started = #i_clk_startTime 1;
	    end
	    else begin
	        i_clk_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.i_clk ) begin
	    if ( i_clk_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( i_clk_started !== 1'b1 )
	        @( i_clk_started );
	
	    i_clk <= i_clk_firstEdge;
	    if ( i_clk_firstHalf > 0.0 ) begin
	        #i_clk_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        i_clk = ~i_clk_firstEdge;
	    end
	    else begin
	        i_clk = ~i_clk_firstEdge;
	    end
	    if ( i_clk_secondHalf > 0.0 ) begin
	        #i_clk_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg i_rst_firstEdge;
	reg i_rst_initialized;
	reg i_rst_started;
	real i_rst_startTime;
	real i_rst_firstHalf;
	real i_rst_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    i_rst_firstEdge = 1'b1;
	    i_rst_startTime = 0.000;		// Offset from time 0 to first edge.
	    i_rst_firstHalf = 5.000;	// Delay for first half of cycle.
	    i_rst_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on i_rst_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( i_rst_initialized !== 1'b1 )
	        @( i_rst_initialized );
	
	    if ( i_rst_startTime > 0.0 ) begin
	        i_rst_started = #i_rst_startTime 1;
	    end
	    else begin
	        i_rst_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.i_rst ) begin
	    if ( i_rst_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( i_rst_started !== 1'b1 )
	        @( i_rst_started );
	
	    i_rst <= i_rst_firstEdge;
	    if ( i_rst_firstHalf > 0.0 ) begin
	        #i_rst_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        i_rst = ~i_rst_firstEdge;
	    end
	    else begin
	        i_rst = ~i_rst_firstEdge;
	    end
	    if ( i_rst_secondHalf > 0.0 ) begin
	        #i_rst_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg i_r_vld_firstEdge;
	reg i_r_vld_initialized;
	reg i_r_vld_started;
	real i_r_vld_startTime;
	real i_r_vld_firstHalf;
	real i_r_vld_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    i_r_vld_firstEdge = 1'b1;
	    i_r_vld_startTime = 0.000;		// Offset from time 0 to first edge.
	    i_r_vld_firstHalf = 5.000;	// Delay for first half of cycle.
	    i_r_vld_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on i_r_vld_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( i_r_vld_initialized !== 1'b1 )
	        @( i_r_vld_initialized );
	
	    if ( i_r_vld_startTime > 0.0 ) begin
	        i_r_vld_started = #i_r_vld_startTime 1;
	    end
	    else begin
	        i_r_vld_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.i_r_vld ) begin
	    if ( i_r_vld_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( i_r_vld_started !== 1'b1 )
	        @( i_r_vld_started );
	
	    i_r_vld <= i_r_vld_firstEdge;
	    if ( i_r_vld_firstHalf > 0.0 ) begin
	        #i_r_vld_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        i_r_vld = ~i_r_vld_firstEdge;
	    end
	    else begin
	        i_r_vld = ~i_r_vld_firstEdge;
	    end
	    if ( i_r_vld_secondHalf > 0.0 ) begin
	        #i_r_vld_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg i_g_vld_firstEdge;
	reg i_g_vld_initialized;
	reg i_g_vld_started;
	real i_g_vld_startTime;
	real i_g_vld_firstHalf;
	real i_g_vld_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    i_g_vld_firstEdge = 1'b1;
	    i_g_vld_startTime = 0.000;		// Offset from time 0 to first edge.
	    i_g_vld_firstHalf = 5.000;	// Delay for first half of cycle.
	    i_g_vld_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on i_g_vld_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( i_g_vld_initialized !== 1'b1 )
	        @( i_g_vld_initialized );
	
	    if ( i_g_vld_startTime > 0.0 ) begin
	        i_g_vld_started = #i_g_vld_startTime 1;
	    end
	    else begin
	        i_g_vld_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.i_g_vld ) begin
	    if ( i_g_vld_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( i_g_vld_started !== 1'b1 )
	        @( i_g_vld_started );
	
	    i_g_vld <= i_g_vld_firstEdge;
	    if ( i_g_vld_firstHalf > 0.0 ) begin
	        #i_g_vld_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        i_g_vld = ~i_g_vld_firstEdge;
	    end
	    else begin
	        i_g_vld = ~i_g_vld_firstEdge;
	    end
	    if ( i_g_vld_secondHalf > 0.0 ) begin
	        #i_g_vld_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg i_b_vld_firstEdge;
	reg i_b_vld_initialized;
	reg i_b_vld_started;
	real i_b_vld_startTime;
	real i_b_vld_firstHalf;
	real i_b_vld_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    i_b_vld_firstEdge = 1'b1;
	    i_b_vld_startTime = 0.000;		// Offset from time 0 to first edge.
	    i_b_vld_firstHalf = 5.000;	// Delay for first half of cycle.
	    i_b_vld_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on i_b_vld_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( i_b_vld_initialized !== 1'b1 )
	        @( i_b_vld_initialized );
	
	    if ( i_b_vld_startTime > 0.0 ) begin
	        i_b_vld_started = #i_b_vld_startTime 1;
	    end
	    else begin
	        i_b_vld_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.i_b_vld ) begin
	    if ( i_b_vld_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( i_b_vld_started !== 1'b1 )
	        @( i_b_vld_started );
	
	    i_b_vld <= i_b_vld_firstEdge;
	    if ( i_b_vld_firstHalf > 0.0 ) begin
	        #i_b_vld_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        i_b_vld = ~i_b_vld_firstEdge;
	    end
	    else begin
	        i_b_vld = ~i_b_vld_firstEdge;
	    end
	    if ( i_b_vld_secondHalf > 0.0 ) begin
	        #i_b_vld_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg o_result_r_busy_firstEdge;
	reg o_result_r_busy_initialized;
	reg o_result_r_busy_started;
	real o_result_r_busy_startTime;
	real o_result_r_busy_firstHalf;
	real o_result_r_busy_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    o_result_r_busy_firstEdge = 1'b1;
	    o_result_r_busy_startTime = 0.000;		// Offset from time 0 to first edge.
	    o_result_r_busy_firstHalf = 5.000;	// Delay for first half of cycle.
	    o_result_r_busy_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on o_result_r_busy_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( o_result_r_busy_initialized !== 1'b1 )
	        @( o_result_r_busy_initialized );
	
	    if ( o_result_r_busy_startTime > 0.0 ) begin
	        o_result_r_busy_started = #o_result_r_busy_startTime 1;
	    end
	    else begin
	        o_result_r_busy_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.o_result_r_busy ) begin
	    if ( o_result_r_busy_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( o_result_r_busy_started !== 1'b1 )
	        @( o_result_r_busy_started );
	
	    o_result_r_busy <= o_result_r_busy_firstEdge;
	    if ( o_result_r_busy_firstHalf > 0.0 ) begin
	        #o_result_r_busy_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        o_result_r_busy = ~o_result_r_busy_firstEdge;
	    end
	    else begin
	        o_result_r_busy = ~o_result_r_busy_firstEdge;
	    end
	    if ( o_result_r_busy_secondHalf > 0.0 ) begin
	        #o_result_r_busy_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg o_result_g_busy_firstEdge;
	reg o_result_g_busy_initialized;
	reg o_result_g_busy_started;
	real o_result_g_busy_startTime;
	real o_result_g_busy_firstHalf;
	real o_result_g_busy_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    o_result_g_busy_firstEdge = 1'b1;
	    o_result_g_busy_startTime = 0.000;		// Offset from time 0 to first edge.
	    o_result_g_busy_firstHalf = 5.000;	// Delay for first half of cycle.
	    o_result_g_busy_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on o_result_g_busy_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( o_result_g_busy_initialized !== 1'b1 )
	        @( o_result_g_busy_initialized );
	
	    if ( o_result_g_busy_startTime > 0.0 ) begin
	        o_result_g_busy_started = #o_result_g_busy_startTime 1;
	    end
	    else begin
	        o_result_g_busy_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.o_result_g_busy ) begin
	    if ( o_result_g_busy_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( o_result_g_busy_started !== 1'b1 )
	        @( o_result_g_busy_started );
	
	    o_result_g_busy <= o_result_g_busy_firstEdge;
	    if ( o_result_g_busy_firstHalf > 0.0 ) begin
	        #o_result_g_busy_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        o_result_g_busy = ~o_result_g_busy_firstEdge;
	    end
	    else begin
	        o_result_g_busy = ~o_result_g_busy_firstEdge;
	    end
	    if ( o_result_g_busy_secondHalf > 0.0 ) begin
	        #o_result_g_busy_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end
	// Variables that can be set externally
	reg o_result_b_busy_firstEdge;
	reg o_result_b_busy_initialized;
	reg o_result_b_busy_started;
	real o_result_b_busy_startTime;
	real o_result_b_busy_firstHalf;
	real o_result_b_busy_secondHalf;
	
	initial begin
	
	    // Only send values from SystemC to the simulator at clock edges.
	    hubSetOption("cycleOnly=1");
	
	    // Default values for config variables.  Values come from static clock.
	    o_result_b_busy_firstEdge = 1'b1;
	    o_result_b_busy_startTime = 0.000;		// Offset from time 0 to first edge.
	    o_result_b_busy_firstHalf = 5.000;	// Delay for first half of cycle.
	    o_result_b_busy_secondHalf = 5.000;	// Second half of cycle less offset value.
	
	    // Make sure that we detect the transition on o_result_b_busy_initalized.
	    hubScheduleDelayedAssignments;
	
	    while ( o_result_b_busy_initialized !== 1'b1 )
	        @( o_result_b_busy_initialized );
	
	    if ( o_result_b_busy_startTime > 0.0 ) begin
	        o_result_b_busy_started = #o_result_b_busy_startTime 1;
	    end
	    else begin
	        o_result_b_busy_started = 1;
	    end
	end
	
	// Make sure that we process delayed assignments after the clock edge.
	always @( SobelFilter0.o_result_b_busy ) begin
	    if ( o_result_b_busy_started === 1'b1 ) begin
	        if ($time == top.cur_time) top.n_cur_time=top.n_cur_time-1;
	        else top.n_cur_time=0;
	        if (top.n_cur_time==0) begin
	            hubScheduleDelayedAssignments;
	        end
	    end
	end
	
	always begin
	    // Make sure our delays are non-zero before starting simulation.
	    while ( o_result_b_busy_started !== 1'b1 )
	        @( o_result_b_busy_started );
	
	    o_result_b_busy <= o_result_b_busy_firstEdge;
	    if ( o_result_b_busy_firstHalf > 0.0 ) begin
	        #o_result_b_busy_firstHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	        o_result_b_busy = ~o_result_b_busy_firstEdge;
	    end
	    else begin
	        o_result_b_busy = ~o_result_b_busy_firstEdge;
	    end
	    if ( o_result_b_busy_secondHalf > 0.0 ) begin
	        #o_result_b_busy_secondHalf;
	        if ($time==top.cur_time) begin
	            top.n_cur_time=top.n_cur_time+1;
	        end else begin
	            top.n_cur_time=1; top.cur_time=$time;
	        end
	    end
	end


endmodule
