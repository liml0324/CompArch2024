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
    input wire [31:0] PC, jal_target, jalr_target, br_target, PC_EX, NPC_EX,    // 这里PC_EX接线时应该接PC_EX-4
    input wire [2:0] br_type_EX,
    input wire jal, jalr, br, pred_err, clk, rst,
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
    reg BTB_history[BTB_SIZE - 1:0];
    wire [31:0] PC_IF;
    integer i;

    

    assign PC_IF = PC - 4;

    

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            for(i = 0; i < BTB_SIZE; i = i + 1) begin
                BTB_target[i] = 0;
                BTB_tag[i] = 0;
                BTB_valid[i] = 0;
                BTB_history[i] = 0;
            end
        end
        else if(br) begin   // 发生了跳转，无论之前是否预测其跳转，都直接在BTB对应位置加上这一表项。这里保证了BTB中只会写入br跳转的表项
            BTB_target[PC_EX[BTB_ADDR_LEN + 1:2]] = br_target;
            BTB_tag[PC_EX[BTB_ADDR_LEN + 1:2]] = PC_EX[31:BTB_ADDR_LEN + 2];
            BTB_valid[PC_EX[BTB_ADDR_LEN + 1:2]] = 1;
            BTB_history[PC_EX[BTB_ADDR_LEN + 1:2]] = 1;
        end
        else if(pred_err && !br)    begin //预测其跳转，实际没跳转（pred_err为1时该指令必然为跳转指令）
            if(BTB_tag[PC_EX[BTB_ADDR_LEN + 1:2]] == PC_EX[31:BTB_ADDR_LEN + 2]) begin
                BTB_history[PC_EX[BTB_ADDR_LEN + 1:2]] = 0;
            end
        end
    end



    always @(*) begin
        if(pred_err)    begin
            if(br)  begin
                NPC = br_target;
            end
            else begin
                NPC = PC_EX + 4;
            end
        end
        if(jalr == 1'b1)    begin
            NPC = jalr_target;
        end
        else if(jal == 1'b1) begin
            NPC = jal_target;
        end
        else if(BTB_valid[PC_IF[BTB_ADDR_LEN + 1:2]] == 1 && BTB_history[PC_IF[BTB_ADDR_LEN + 1:2]] == 1 && BTB_tag[PC_IF[BTB_ADDR_LEN + 1:2]] == PC_IF[31:BTB_ADDR_LEN + 2]) begin
            NPC = BTB_target[PC_IF[BTB_ADDR_LEN + 1:2]];
        end
        else begin
            NPC = PC;
        end
    end

endmodule