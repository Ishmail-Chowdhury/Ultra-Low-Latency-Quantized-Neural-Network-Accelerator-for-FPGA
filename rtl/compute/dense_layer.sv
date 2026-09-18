`timescale 1ns/1ps
`default_nettype none

// One vector in flight, LANES parallel neurons, static row-major weights.
module dense_layer #(
    parameter integer INPUTS = 8,
    parameter integer OUTPUTS = 4,
    parameter integer ACC_WIDTH = 32,
    parameter integer SHIFT = 7,
    parameter bit RELU = 1'b0,
    parameter WEIGHT_FILE = "weights.mem",
    parameter BIAS_FILE = "biases.mem",
    parameter integer LANES = 1
) (
    input  wire                    clk,
    input  wire                    rst,
    input  wire signed [7:0]       s_tdata,
    input  wire                    s_tvalid,
    output wire                    s_tready,
    input  wire                    s_tlast,
    output logic signed [7:0]      m_tdata,
    output wire                    m_tvalid,
    input  wire                    m_tready,
    output wire                    m_tlast
);
    localparam integer IN_INDEX_WIDTH = INPUTS > 1 ? $clog2(INPUTS) : 1;
    localparam integer OUT_INDEX_WIDTH = OUTPUTS > 1 ? $clog2(OUTPUTS) : 1;
    localparam integer LANE_INDEX_WIDTH = LANES > 1 ? $clog2(LANES) : 1;
    typedef enum logic [2:0] {RECEIVE, LOAD_BIAS, ACCUMULATE, QUANTIZE, SEND} state_t;
    state_t state;
    logic [IN_INDEX_WIDTH-1:0] input_index;
    logic [OUT_INDEX_WIDTH-1:0] output_index;
    logic [OUT_INDEX_WIDTH-1:0] group_base;
    logic [LANE_INDEX_WIDTH-1:0] send_lane;
    logic signed [7:0] inputs [0:INPUTS-1];
    logic signed [7:0] weights [0:INPUTS*OUTPUTS-1];
    logic signed [ACC_WIDTH-1:0] biases [0:OUTPUTS-1];
    logic signed [7:0] results [0:LANES-1];
    wire signed [7:0] quantized [0:LANES-1];
    integer lane;

    // Each active lane needs an asynchronous weight read. Resource mapping and
    // timing require synthesis; this is not a synchronous BRAM implementation.
    initial begin
        $readmemh(WEIGHT_FILE, weights);
        $readmemh(BIAS_FILE, biases);
    end

    assign s_tready = !rst && state == RECEIVE;
    assign m_tvalid = !rst && state == SEND;
    assign m_tlast = m_tvalid && output_index == OUTPUTS - 1;
    assign m_tdata = results[send_lane];

    for (genvar g = 0; g < LANES; g = g + 1) begin : lanes
        wire active_lane = (group_base + g) < OUTPUTS;
        wire signed [ACC_WIDTH-1:0] accumulator;
        // Mask padding in the last group: never accumulate uninitialized rows.
        wire signed [7:0] weight = active_lane
            ? weights[(group_base + g) * INPUTS + input_index] : 8'sd0;
        wire signed [ACC_WIDTH-1:0] bias = active_lane ? biases[group_base + g] : '0;
        mac #(.ACC_WIDTH(ACC_WIDTH)) compute (
            .clk(clk), .rst(rst), .load(state == LOAD_BIAS),
            .enable(state == ACCUMULATE && active_lane), .x(inputs[input_index]),
            .weight(weight), .bias(bias), .acc(accumulator)
        );
        requantize #(.ACC_WIDTH(ACC_WIDTH), .SHIFT(SHIFT), .RELU(RELU)) post (
            .acc(accumulator), .result(quantized[g])
        );
    end

    always_ff @(posedge clk) begin
        if (rst) begin
            state <= RECEIVE;
            input_index <= '0;
            output_index <= '0;
            group_base <= '0;
            send_lane <= '0;
            for (lane = 0; lane < LANES; lane = lane + 1)
                results[lane] <= '0;
        end else begin
            case (state)
                RECEIVE: if (s_tvalid && s_tready) begin
                    inputs[input_index] <= s_tdata;
                    if (input_index == INPUTS - 1) begin
                        input_index <= '0;
                        output_index <= '0;
                        group_base <= '0;
                        state <= LOAD_BIAS;
                    end else
                        input_index <= input_index + 1'b1;
                end
                LOAD_BIAS: begin
                    input_index <= '0;
                    state <= ACCUMULATE;
                end
                ACCUMULATE: begin
                    if (input_index == INPUTS - 1)
                        state <= QUANTIZE;
                    else
                        input_index <= input_index + 1'b1;
                end
                QUANTIZE: begin
                    for (lane = 0; lane < LANES; lane = lane + 1)
                        results[lane] <= quantized[lane];
                    send_lane <= '0;
                    state <= SEND;
                end
                SEND: if (m_tvalid && m_tready) begin
                    if (output_index == OUTPUTS - 1) begin
                        input_index <= '0;
                        output_index <= '0;
                        state <= RECEIVE;
                    end else begin
                        output_index <= output_index + 1'b1;
                        if (send_lane == LANES - 1) begin
                            group_base <= output_index + 1'b1;
                            state <= LOAD_BIAS;
                        end else
                            send_lane <= send_lane + 1'b1;
                    end
                end
                default: begin
                    state <= RECEIVE;
                    input_index <= '0;
                    output_index <= '0;
                    group_base <= '0;
                    send_lane <= '0;
                end
            endcase
        end
    end

`ifndef SYNTHESIS
    logic stalled;
    logic signed [7:0] held_data;
    logic held_last;
    initial begin
        if (INPUTS < 1 || OUTPUTS < 1)
            $fatal(1, "Dense dimensions must be positive");
        if (LANES < 1 || LANES > OUTPUTS)
            $fatal(1, "LANES must be between 1 and OUTPUTS");
    end
    always @(posedge clk) begin
        if (rst) begin
            stalled <= 1'b0;
        end else begin
            if (s_tvalid && s_tready) begin
                if (s_tlast !== (input_index == INPUTS - 1))
                    $fatal(1, "Input tlast does not match vector length");
                if ($isunknown(s_tdata))
                    $fatal(1, "Unknown input data");
            end
            if (stalled && (!m_tvalid || m_tdata !== held_data || m_tlast !== held_last))
                $fatal(1, "Output changed under backpressure");
            stalled <= m_tvalid && !m_tready;
            held_data <= m_tdata;
            held_last <= m_tlast;
        end
    end
`endif
endmodule

`default_nettype wire
