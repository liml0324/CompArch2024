`timescale 1ns / 1ps
//  功能说明
    //  根据跳转信号，决定执行的下一条指令地址
    //  debug端口用于simulation时批量写入数据，可以忽略
// 输入
    // PC                指令地址（PC + 4, 而非PC）
    // jal_target        jal跳转地址
    // jalr_target       jalr跳转地址
    // br_target         br跳转地址
    // jal               jal == 1时，有jal跳转
    // jalr              jalr == 1时，有jalr跳转
    // br                br == 1时，有br跳转
// 输出
    // NPC               下一条执行的指令地址
// 实验要求  
    // 实现NPC_Generator

module NPC_Generator(
    input wire [31:0] PC, jal_target, jalr_target, br_target,
    input wire jal, jalr, br, pred_err,
    output reg [31:0] NPC
    );

    // TODO: Complete this module

    /* FIXME: Write your code here... */

    localparam BTB_ADDR_LEN = 6;
    localparam BTB_SIZE = 1 << BTB_ADDR_LEN;
    localparam BTB_TAG_LEN = 30 - BTB_ADDR_LEN; // 指令地址是字对齐的，所以低两位是0，不需要考虑

    reg [31:0] BTB_target[BTB_SIZE - 1:0];
    reg [BTB_TAG_LEN - 1:0] BTB_tag[BTB_SIZE - 1:0];
    reg BTB_valid[BTB_SIZE - 1:0];

    always @(*) begin
        if(jalr == 1'b1)    begin
            NPC = jalr_target;
        end
        else if(br == 1'b1) begin
            NPC = br_target;
        end
        else if(jal == 1'b1) begin
            NPC = jal_target;
        end
        else begin
            NPC = PC;
        end
    end

endmodule