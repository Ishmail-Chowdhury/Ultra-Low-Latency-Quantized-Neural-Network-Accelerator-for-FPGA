`timescale 1ns/1ps
`default_nettype none

// Bias-first signed accumulator. Priority: reset > load > enable > hold.
module mac #(
    parameter integer DATA_WIDTH = 8,
    parameter integer ACC_WIDTH = 32
) (
    input  wire                         clk,
    input  wire                         rst,
    input  wire                         load,
    input  wire                         enable,
    input  wire signed [DATA_WIDTH-1:0]  x,
    input  wire signed [DATA_WIDTH-1:0]  weight,
    input  wire signed [ACC_WIDTH-1:0]   bias,
    output logic signed [ACC_WIDTH-1:0]  acc
);
    localparam integer PRODUCT_WIDTH = 2 * DATA_WIDTH;
    wire signed [PRODUCT_WIDTH-1:0] product = x * weight;
    wire signed [ACC_WIDTH:0] extended_product =
        {{(ACC_WIDTH + 1 - PRODUCT_WIDTH){product[PRODUCT_WIDTH-1]}}, product};
    wire signed [ACC_WIDTH:0] sum = $signed({acc[ACC_WIDTH-1], acc}) + extended_product;

    always_ff @(posedge clk) begin
        if (rst)
            acc <= '0;
        else if (load)
            acc <= bias;
        else if (enable)
            acc <= sum[ACC_WIDTH-1:0];
    end

`ifndef SYNTHESIS
    initial begin
        if (DATA_WIDTH < 2 || ACC_WIDTH < PRODUCT_WIDTH)
            $fatal(1, "MAC requires DATA_WIDTH >= 2 and ACC_WIDTH >= 2*DATA_WIDTH");
    end
    always @(posedge clk) begin
        if (!rst && !load && enable) begin
            if ((sum[ACC_WIDTH] !== sum[ACC_WIDTH-1]) || $isunknown(sum))
                $fatal(1, "MAC accumulator overflow or unknown arithmetic");
        end
    end
`endif
endmodule

`default_nettype wire
