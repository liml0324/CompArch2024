`timescale 1ns / 1ps
//  功能说明
    // IF\ID的PC段寄存器
// 输入
    // clk               时钟信号
    // PC_IF             PC寄存器传来的指令地址
    // bubbleD           ID阶段的bubble信号
    // flushD            ID阶段的flush信号
// 输出
    // PC_ID             传给下一段寄存器的PC地址
// 实验要求  
    // 无需修改

module PC_ID(
    input wire clk, bubbleD, flushD,
    input wire [31:0] PC_IF, NPC_IF,
    output reg [31:0] PC_ID, NPC_ID
    );

    initial PC_ID = 0;
    initial NPC_ID = 0;
    
    always@(posedge clk)
        if (!bubbleD) 
        begin
            if (flushD) begin
                PC_ID <= 0;
                NPC_ID <= 0;
            end
            else    begin
                PC_ID <= PC_IF;
                NPC_ID <= NPC_IF;
            end
        end
    
endmodule