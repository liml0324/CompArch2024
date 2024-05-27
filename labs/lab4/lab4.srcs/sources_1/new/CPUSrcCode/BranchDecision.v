`timescale 1ns / 1ps
//  功能说明
    //  判断是否branch
// 输入
    // reg1               寄存器1
    // reg2               寄存器2
    // br_type            branch类型
// 输出
    // br                 是否branch
// 实验要求
    // 补全模块

`include "Parameters.v"   
module BranchDecision(
    input wire [31:0] reg1, reg2, PC_EX_4, NPC_EX, br_target,
    input wire [2:0] br_type,
    input wire clk, rst, bubbleE,
    output reg br, pred_err
    );

    // TODO: Complete this module

    reg [31:0] success_pred, fail_pred, cycles;

    always @(posedge clk or posedge rst) begin
        if(rst) begin
            success_pred <= 0;
            fail_pred <= 0;
            cycles <= 0;
        end
        else if(|br_type && !bubbleE) begin     // 有branch指令，且不是bubble
            if(pred_err)    begin
                fail_pred <= fail_pred + 1;
            end
            else begin
                success_pred <= success_pred + 1;
            end
            cycles = cycles + 1;
        end
    end

    always @ (*)
    begin
        case(br_type)
            `NOBRANCH: br = 0;
            `BEQ: br = (reg1 == reg2) ? 1 : 0;
            `BLTU: br = (reg1 < reg2) ? 1 : 0;

            /* FIXME: Write your code here... */
            `BNE: br = (reg1 != reg2) ? 1 : 0;
            `BLT: br = ($signed(reg1) < $signed(reg2)) ? 1 : 0;
            `BGE: br = ($signed(reg1) >= $signed(reg2)) ? 1 : 0;
            `BGEU: br = (reg1 >= reg2) ? 1 : 0;

            default: br = 0;
        endcase
    end

    always @ (*)    begin
        if(br && NPC_EX != br_target)   begin
            pred_err = 1;
        end
        else if(!br && NPC_EX != PC_EX_4)    begin
            pred_err = 1;
        end
        else    begin
            pred_err = 0;
        end
    end

endmodule
