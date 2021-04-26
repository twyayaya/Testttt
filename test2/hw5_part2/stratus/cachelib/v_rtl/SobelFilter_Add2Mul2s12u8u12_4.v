`timescale 1ps / 1ps
/*****************************************************************************
    Verilog RTL Description
    
    Configured at: 12:57:57 CST (+0800), Monday 26 April 2021
    Configured on: ws32
    Configured by: m109061613 (m109061613)
    
    Created by: Stratus DpOpt 2019.1.01 
*******************************************************************************/

module SobelFilter_Add2Mul2s12u8u12_4 (
	in3,
	in2,
	in1,
	out1
	); /* architecture "behavioural" */ 
input [11:0] in3;
input [7:0] in2;
input [11:0] in1;
output [11:0] out1;
wire [11:0] asc001;

wire [11:0] asc001_tmp_0;
assign asc001_tmp_0 = 
	+(in3);
assign asc001 = asc001_tmp_0
	+(in1 * in2);

assign out1 = asc001;
endmodule

/* CADENCE  uLL5SAg= : u9/ySgnWtBlWxVPRXgAZ4Og= ** DO NOT EDIT THIS LINE ******/


