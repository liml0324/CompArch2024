// `define LRU

module cache #(
    parameter  LINE_ADDR_LEN = 2, // line内地址长度，决定了每个line具有2^3个word
    parameter  SET_ADDR_LEN  = 4, // 组地址长度，决定了一共有2^3=8组
    parameter  TAG_ADDR_LEN  = 5, // tag长度
    parameter  WAY_CNT       = 3  // 组相连度，决定了每组中有多少路line，这里是直接映射型cache，因此该参数没用到
)(
    input  clk, rst,
    output miss,               // 对CPU发出的miss信号
    input  [31:0] addr,        // 读写请求地址
    input  rd_req,             // 读请求信号
    output reg [31:0] rd_data, // 读出的数据，一次读一个word
    input  wr_req,             // 写请求信号
    input  [31:0] wr_data      // 要写入的数据，一次写一个word
);

localparam MEM_ADDR_LEN    = TAG_ADDR_LEN + SET_ADDR_LEN ; // 计算主存地址长度 MEM_ADDR_LEN，主存大小=2^MEM_ADDR_LEN个line
localparam UNUSED_ADDR_LEN = 32 - TAG_ADDR_LEN - SET_ADDR_LEN - LINE_ADDR_LEN - 2 ;       // 计算未使用的地址的长度

localparam LINE_SIZE       = 1 << LINE_ADDR_LEN  ;         // 计算 line 中 word 的数量，即 2^LINE_ADDR_LEN 个word 每 line
localparam SET_SIZE        = 1 << SET_ADDR_LEN   ;         // 计算一共有多少组，即 2^SET_ADDR_LEN 个组

localparam WAY_ADDR_LEN    = WAY_CNT > 1 ? $clog2(WAY_CNT) : 1 ;             // 计算路号的长度，即 3个路，需要2bit来表示

reg [            31:0] cache_mem    [WAY_CNT][SET_SIZE][LINE_SIZE]; // SET_SIZE个line，每个line有LINE_SIZE个word
reg [TAG_ADDR_LEN-1:0] cache_tags   [WAY_CNT][SET_SIZE];            // SET_SIZE个TAG
reg                    valid        [WAY_CNT][SET_SIZE];            // SET_SIZE个valid(有效位)
reg                    dirty        [WAY_CNT][SET_SIZE];            // SET_SIZE个dirty(脏位)

reg [WAY_CNT-1:0]  hit_way;        // 命中的路号，用独热码表示
reg [WAY_ADDR_LEN-1:0] way_addr;   // 路地址


wire [              2-1:0]   word_addr;                   // 将输入地址addr拆分成这5个部分
wire [  LINE_ADDR_LEN-1:0]   line_addr;
wire [   SET_ADDR_LEN-1:0]    set_addr;
wire [   TAG_ADDR_LEN-1:0]    tag_addr;
wire [UNUSED_ADDR_LEN-1:0] unused_addr;

enum  {IDLE, SWAP_OUT, SWAP_IN, SWAP_IN_OK} cache_stat;    // cache 状态机的状态定义
                                                           // IDLE代表就绪，SWAP_OUT代表正在换出，SWAP_IN代表正在换入，SWAP_IN_OK代表换入后进行一周期的写入cache操作。

reg  [   SET_ADDR_LEN-1:0] mem_rd_set_addr = 0;
reg  [   TAG_ADDR_LEN-1:0] mem_rd_tag_addr = 0;
wire [   MEM_ADDR_LEN-1:0] mem_rd_addr = {mem_rd_tag_addr, mem_rd_set_addr};
reg  [   MEM_ADDR_LEN-1:0] mem_wr_addr = 0;

reg  [31:0] mem_wr_line [LINE_SIZE];
wire [31:0] mem_rd_line [LINE_SIZE];

wire mem_gnt;      // 主存响应读写的握手信号

reg [31:0] access_times;
reg [31:0] miss_times;
reg [31:0] access_cycles;

assign {unused_addr, tag_addr, set_addr, line_addr, word_addr} = addr;  // 拆分 32bit ADDR

always @(posedge clk or posedge rst) begin
    if(rst) access_cycles <= 0;
    else if(rd_req | wr_req)    access_cycles <= access_cycles + 1;
end

reg [WAY_ADDR_LEN-1:0] onehot2bin [1 << WAY_CNT];  // 独热码转二进制码
initial begin
    // integer j;
    if(WAY_CNT > 1) begin
        for(integer i = 1, j = 0; i < (1 << WAY_CNT); i = i << 1, j++) begin
            // j = $clog2(i);
            onehot2bin[i] = j[WAY_ADDR_LEN-1:0];
        end
    end
    
end

always @(*) begin
    for(integer i = 0; i < WAY_CNT; i++) begin
        if(valid[i][set_addr] && cache_tags[i][set_addr] == tag_addr) begin
            hit_way[i] = 1'b1;
        end 
        else begin
            hit_way[i] = 1'b0;
        end
    end
end


reg cache_hit = 1'b0;
always @ (*) begin              // 判断 输入的address 是否在 cache 中命中
    if(|hit_way)   // 如果有一路命中，则命中
        cache_hit = 1'b1;
    else
        cache_hit = 1'b0;
end

`ifdef LRU
reg [WAY_ADDR_LEN-1:0] lru [WAY_CNT][SET_SIZE];   // LRU算法中，每个组中的每一路对应一个WAY_ADDR_LEN位的LRU计数器
reg [WAY_ADDR_LEN-1:0] lru_way[SET_SIZE];         // 下一次miss时替换的路号
reg [WAY_CNT-1:0]      lru_onehot;                // 用独热码表示下一次miss时应该替换哪路
always @(*) begin
    for(integer i = 0; i < WAY_CNT; i++) begin
        if(lru[way_addr][set_addr] == 0 && lru[i][set_addr] == 1)   begin
            lru_onehot[i] = 1'b1;
        end
        else if(lru[way_addr][set_addr] != 0 && lru[i][set_addr] == 0)  begin
            lru_onehot[i] = 1'b1;
        end
        else    begin
            lru_onehot[i] = 1'b0;
        end
    end
end
always @(posedge clk or posedge rst) begin
    if(rst) begin
        for(integer i = 0; i < SET_SIZE; i++) begin
            for(integer j = 0; j < WAY_CNT; j++) begin
                lru[j][i] <= j[WAY_ADDR_LEN-1:0];
            end
            
            lru_way[i] <= 0;
        end
    end
    else if(cache_stat == IDLE && cache_hit && (wr_req | rd_req)) begin
        for(integer i = 0; i < WAY_CNT; i++) begin
            if(lru[i][set_addr] > lru[way_addr][set_addr]) begin    // 如果有路的LRU计数器比命中的路的大，则减1，相当于整体往前移动
                lru[i][set_addr] <= lru[i][set_addr] - 1;
            end
        end
        lru[way_addr][set_addr] <= WAY_CNT-1;   // 命中的路号置为最大值
        lru_way[set_addr] <= onehot2bin[lru_onehot];
    end
end
`else
reg [WAY_ADDR_LEN-1:0] fifo [WAY_CNT][SET_SIZE];  
reg [WAY_ADDR_LEN-1:0] fifo_way[SET_SIZE];
reg [WAY_CNT-1:0]      fifo_onehot;
always @(*) begin
    for(integer i = 0; i < WAY_CNT; i++) begin
        if(fifo[i][set_addr] == 1)  begin
            fifo_onehot[i] = 1'b1;
        end
        else    begin
            fifo_onehot[i] = 1'b0;
        end
    end
end
always @(posedge clk or posedge rst)    begin
    if(rst) begin
        for(integer i = 0; i < SET_SIZE; i++)   begin
            for(integer j = 0; j < WAY_CNT; j++) begin
                fifo[j][i] <= j[WAY_ADDR_LEN-1:0];
            end

            fifo_way[i] <= 0;
        end
    end
    else if(cache_stat == SWAP_IN_OK)   begin
        for(integer i = 0; i < WAY_CNT; i++)    begin
            if(fifo[i][set_addr] > 0)   fifo[i][set_addr] <= fifo[i][set_addr] - 1;
            else    fifo[i][set_addr] <= WAY_CNT - 1;
        end
        fifo_way[set_addr] <= onehot2bin[fifo_onehot];
    end
end
`endif

always @(*) begin
    if(WAY_CNT == 1)    begin
        way_addr = 0;
    end
    else if(cache_stat == IDLE && cache_hit) begin // 如果命中，则将命中的路号转换为二进制码
        way_addr = onehot2bin[hit_way]; 
    end 
    else begin  // 如果未命中，需要根据换出策略选择被换出的地址
        // way_addr = 0;
        `ifdef LRU
        way_addr = lru_way[set_addr];
        `else
        way_addr = fifo_way[set_addr];
        `endif
    end
end


always @ (posedge clk or posedge rst) begin     // ?? cache ???
    if(rst) begin
        cache_stat <= IDLE;
        for(integer i = 0; i < SET_SIZE; i++) begin
            for(integer j = 0; j < WAY_CNT; j++) begin
                valid     [j][i] <= 0;
                dirty     [j][i] <= 0;
            end
        end
        for(integer k = 0; k < LINE_SIZE; k++)
            mem_wr_line[k] <= 0;
        mem_wr_addr <= 0;
        {mem_rd_tag_addr, mem_rd_set_addr} <= 0;
        rd_data <= 0;
        access_times <= 0;
        miss_times <= 0;
    end else begin
        case(cache_stat)
        IDLE:       begin
                        if(cache_hit) begin
                            if(rd_req) begin    // 如果cache命中，并且是读请求，
                                access_times <= access_times + 1;
                                rd_data <= cache_mem[way_addr][set_addr][line_addr];   //则直接从cache中取出要读的数据
                            end else if(wr_req) begin // 如果cache命中，并且是写请求，
                                access_times <= access_times + 1;
                                cache_mem[way_addr][set_addr][line_addr] <= wr_data;   // 则直接向cache中写入数据
                                dirty[way_addr][set_addr] <= 1'b1;                     // 写数据的同时置脏位
                            end 
                        end else begin
                            if(wr_req | rd_req) begin   // 如果 cache 未命中，并且有读写请求，则需要进行换入
                                miss_times <= miss_times + 1;
                                if(valid[way_addr][set_addr] & dirty[way_addr][set_addr]) begin    // 如果 要换入的cache line 本来有效，且脏，则需要先将它换出
                                    cache_stat  <= SWAP_OUT;
                                    mem_wr_addr <= {cache_tags[way_addr][set_addr], set_addr};
                                    mem_wr_line <= cache_mem[way_addr][set_addr];
                                end else begin                                   // 反之，不需要换出，直接换入
                                    cache_stat  <= SWAP_IN;
                                end
                                {mem_rd_tag_addr, mem_rd_set_addr} <= {tag_addr, set_addr};
                            end
                        end
                    end
        SWAP_OUT:   begin
                        if(mem_gnt) begin           // 如果主存握手信号有效，说明换出成功，跳到下一状态
                            cache_stat <= SWAP_IN;
                        end
                    end
        SWAP_IN:    begin
                        if(mem_gnt) begin           // 如果主存握手信号有效，说明换入成功，跳到下一状态
                            cache_stat <= SWAP_IN_OK;
                        end
                    end
        SWAP_IN_OK: begin           // 上一个周期换入成功，这周期将主存读出的line写入cache，并更新tag，置高valid，置低dirty
                        for(integer i=0; i<LINE_SIZE; i++)  cache_mem[way_addr][mem_rd_set_addr][i] <= mem_rd_line[i];
                        cache_tags[way_addr][mem_rd_set_addr] <= mem_rd_tag_addr;
                        valid     [way_addr][mem_rd_set_addr] <= 1'b1;
                        dirty     [way_addr][mem_rd_set_addr] <= 1'b0;
                        cache_stat <= IDLE;        // 回到就绪状态
                    end
        endcase
    end
end

wire mem_rd_req = (cache_stat == SWAP_IN );
wire mem_wr_req = (cache_stat == SWAP_OUT);
wire [   MEM_ADDR_LEN-1 :0] mem_addr = mem_rd_req ? mem_rd_addr : ( mem_wr_req ? mem_wr_addr : 0);

assign miss = (rd_req | wr_req) & ~(cache_hit && cache_stat==IDLE) ;     // 当 有读写请求时，如果cache不处于就绪(IDLE)状态，或者未命中，则miss=1

main_mem #(     // 主存，每次读写以line 为单位
    .LINE_ADDR_LEN  ( LINE_ADDR_LEN          ),
    .ADDR_LEN       ( MEM_ADDR_LEN           )
) main_mem_instance (
    .clk            ( clk                    ),
    .rst            ( rst                    ),
    .gnt            ( mem_gnt                ),
    .addr           ( mem_addr               ),
    .rd_req         ( mem_rd_req             ),
    .rd_line        ( mem_rd_line            ),
    .wr_req         ( mem_wr_req             ),
    .wr_line        ( mem_wr_line            )
);

endmodule





