`timescale 1ns/1ps
`default_nettype none

// Two static dense layers connected directly by ready/valid flow control.
module mlp #(
    parameter integer INPUTS = 64,
    parameter integer HIDDEN = 16,
    parameter integer OUTPUTS = 10,
    parameter integer ACC_WIDTH = 32,
    parameter integer HIDDEN_SHIFT = 7,
    parameter integer OUTPUT_SHIFT = 7,
    parameter HIDDEN_WEIGHTS = "layer0/weights.mem",
    parameter HIDDEN_BIASES = "layer0/biases.mem",
    parameter OUTPUT_WEIGHTS = "layer1/weights.mem",
    parameter OUTPUT_BIASES = "layer1/biases.mem",
    parameter integer HIDDEN_LANES = 1,
    parameter integer OUTPUT_LANES = 1
) (
    input  wire               clk,
    input  wire               rst,
    input  wire signed [7:0]  s_tdata,
    input  wire               s_tvalid,
    output wire               s_tready,
    input  wire               s_tlast,
    output wire signed [7:0] m_tdata,
    output wire               m_tvalid,
    input  wire               m_tready,
    output wire               m_tlast
);
    wire signed [7:0] hidden_data;
    wire hidden_valid, hidden_ready, hidden_last;

    dense_layer #(
        .INPUTS(INPUTS), .OUTPUTS(HIDDEN), .ACC_WIDTH(ACC_WIDTH),
        .SHIFT(HIDDEN_SHIFT), .RELU(1'b1), .LANES(HIDDEN_LANES),
        .WEIGHT_FILE(HIDDEN_WEIGHTS), .BIAS_FILE(HIDDEN_BIASES)
    ) hidden_layer (
        .clk(clk), .rst(rst), .s_tdata(s_tdata), .s_tvalid(s_tvalid),
        .s_tready(s_tready), .s_tlast(s_tlast),
        .m_tdata(hidden_data), .m_tvalid(hidden_valid),
        .m_tready(hidden_ready), .m_tlast(hidden_last)
    );

    dense_layer #(
        .INPUTS(HIDDEN), .OUTPUTS(OUTPUTS), .ACC_WIDTH(ACC_WIDTH),
        .SHIFT(OUTPUT_SHIFT), .RELU(1'b0), .LANES(OUTPUT_LANES),
        .WEIGHT_FILE(OUTPUT_WEIGHTS), .BIAS_FILE(OUTPUT_BIASES)
    ) output_layer (
        .clk(clk), .rst(rst), .s_tdata(hidden_data), .s_tvalid(hidden_valid),
        .s_tready(hidden_ready), .s_tlast(hidden_last),
        .m_tdata(m_tdata), .m_tvalid(m_tvalid),
        .m_tready(m_tready), .m_tlast(m_tlast)
    );
endmodule

`default_nettype wire
