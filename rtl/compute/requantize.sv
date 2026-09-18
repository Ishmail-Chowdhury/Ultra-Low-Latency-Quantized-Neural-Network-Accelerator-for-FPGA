`timescale 1ns/1ps
`default_nettype none

// Static power-of-two scaling, nearest rounding with ties away from zero.
module requantize #(
    parameter integer ACC_WIDTH = 32,
    parameter integer SHIFT = 7,
    parameter bit RELU = 1'b0
) (
    input  wire signed [ACC_WIDTH-1:0] acc,
    output logic signed [7:0]         result
);
    wire signed [ACC_WIDTH:0] extended_acc = {acc[ACC_WIDTH-1], acc};
    wire [ACC_WIDTH:0] magnitude = acc[ACC_WIDTH-1] ? -extended_acc : extended_acc;
    wire [ACC_WIDTH:0] rounded_magnitude;
    wire signed [ACC_WIDTH:0] rounded = acc[ACC_WIDTH-1]
        ? -$signed(rounded_magnitude) : $signed(rounded_magnitude);

    generate
        if (SHIFT == 0) begin : no_shift
            assign rounded_magnitude = magnitude;
        end else begin : round_shift
            localparam [ACC_WIDTH:0] HALF = {{ACC_WIDTH{1'b0}}, 1'b1} << (SHIFT - 1);
            assign rounded_magnitude = (magnitude + HALF) >> SHIFT;
        end
    endgenerate

    always_comb begin
        if (RELU && rounded < 0)
            result = 8'sd0;
        else if (rounded > 127)
            result = 8'sd127;
        else if (rounded < -128)
            result = 8'sh80;
        else
            result = rounded[7:0];
    end

`ifndef SYNTHESIS
    initial begin
        if (ACC_WIDTH < 16 || SHIFT < 0 || SHIFT >= ACC_WIDTH)
            $fatal(1, "Invalid requantization width or shift");
    end
`endif
endmodule

`default_nettype wire
