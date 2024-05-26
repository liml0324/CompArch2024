`timescale 1ns / 1ps
// 实验要求
    // 补全模块（阶段三）

module CSR_Regfile(
    input wire clk,
    input wire rst,
    input wire CSR_write_en,
    input wire [11:0] CSR_write_addr,
    input wire [11:0] CSR_read_addr,
    input wire [31:0] CSR_data_write,
    output wire [31:0] CSR_data_read
    );
    
    // TODO: Complete this module

    /* FIXME: Write your code here... */
    integer i;
    reg [31:0] CSR_regfile [0:4095];

    assign CSR_data_read = CSR_regfile[CSR_read_addr];

    always @(posedge clk) begin
        if (rst) begin
            for (i = 0; i < 4096; i = i + 1) begin
                CSR_regfile[i] <= 32'b0;
            end
        end
        else if (CSR_write_en) begin
            CSR_regfile[CSR_write_addr] <= CSR_data_write;
        end
    end

endmodule
