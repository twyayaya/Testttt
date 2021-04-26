`timescale 1ps / 1ps
/*****************************************************************************
    Verilog RTL Description
    
    Configured at: 12:38:38 CST (+0800), Monday 26 April 2021
    Configured on: ws32
    Configured by: m109061613 (m109061613)
    
    Created by: Stratus DpOpt 2019.1.01 
*******************************************************************************/

module SobelFilter_DECODE_16U_2_4 (
	in1,
	out1
	); /* architecture "behavioural" */ 
input [3:0] in1;
output [15:0] out1;
wire [15:0] asc001;

assign asc001 = 16'B0000000000000001 << in1;

assign out1 = asc001;
endmodule

/* CADENCE  ubD3TAw= : u9/ySgnWtBlWxVPRXgAZ4Og= ** DO NOT EDIT THIS LINE ******/


