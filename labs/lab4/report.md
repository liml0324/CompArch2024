# Lab 4 Report
## 代码实现
### BTB
由于分支预测结果需要在IF阶段给出，因此我修改了`NPCGenerator.v`，在其中添加BTB相关的逻辑。

首先，我在其中加入了下面几组寄存器：
```verilog
localparam BTB_ADDR_LEN = 6;
localparam BTB_SIZE = 1 << BTB_ADDR_LEN;
localparam BTB_TAG_LEN = 30 - BTB_ADDR_LEN; // 指令地址是字对齐的，所以低两位是0，不需要考虑

reg [31:0] BTB_target[BTB_SIZE - 1:0];  // BTB的目标地址
reg [BTB_TAG_LEN - 1:0] BTB_tag[BTB_SIZE - 1:0];
reg BTB_valid[BTB_SIZE - 1:0];  // BTB的有效位
reg BTB_history[BTB_SIZE - 1:0];    // BTB的预测位
```
其中`BTB_ADDR_LEN`是BTB的索引长度，可以用来控制BTB的大小。

另外，还需要修改`NPCGenerator`的输入输出端口，将EX阶段的`PC`、`NPC`和`br_type`接入。由于增加了存储器，还需接入时钟和复位信号。为方便计算分支预测效果和时钟周期数，还需接入`bubbleE`信号。另外，增加`pred_err`信号，表示分支预测错误，传递给`Harzard`模块冲刷流水线。

BTB的预测逻辑为：在IF阶段，根据索引（PC低6位）在BTB中找到对应表项，如果tag相同、BTB有效且history位为1（表示跳转）则将BTB的目标地址作为`NPC`，否则将`PC+4`作为`NPC`。这一部分由组合电路完成。

BTB的更新逻辑如下：
- 复位时，将所有BTB存储器清零。
- 如果发生了跳转，将`br_target`写入`BTB_target`对应位置，对应指令PC的高位写入`BTB_tag`，`BTB_valid`置1，`BTB_history`置1。这里如果预测正确，可以不用写入，但为了简化代码逻辑，每次都写入。
- 如果出现预测跳转，实际未跳转的情况，将对应位置的`BTB_history`置0。

### BHT
在BTB的基础上，增加BHT相关表项：
```verilog
localparam BHT_ADDR_LEN = 12;   // 4096项BHT
localparam BHT_SIZE = 1 << BHT_ADDR_LEN;

reg [1:0] BHT_history[BHT_SIZE - 1:0];
```
`BHT_history`有2位，共4个状态。其中`2'b00`, `2'b01`表示跳转，`2'b10`, `2'b11`表示不跳转。

BHT的预测逻辑为，只要BTB命中（BTB中的tag和PC高位相同，且valid位为1），且BHT的高位为0，则将`NPC`设为`BTB_target`，否则为`PC+4`。这一部分由组合电路完成。

BHT的更新逻辑为：
- 只要EX阶段实际发生跳转，且BHT对应表项大于0，则将BHT对应表项减1
- 如果EX阶段实际不跳转，且BHT对应表项小于3，则将BHT对应表项加1

### 预测情况统计
使用下面3个寄存器进行统计：
```verilog
reg [31:0] fail_pred, totol_pred, cycles;
```
`fail_pred`表示预测错误的次数，`totol_pred`表示总预测次数，`cycles`表示总时钟周期数。

复位时将这3个寄存器清零，每次发生预测错误且`bubbleE`不为1时，`fail_pred`加1；`br_type_EX`不为1且`bubbleE`不为1时，`totol_pred`加1；每个`bubbleE`不为1的时钟周期，`cycles`加1。

### Branch History Table
文档中要求补全的表格：
|BTB|BHT|REAL|NPC_PRED|flush|NPC_REAL|BTB_update|
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
|Y|Y|Y|BUF|N|BUF(=BR_TARGET)|N|
|Y|Y|N|BUF|Y|PC_EX+4|N|
|Y|N|Y|PC_IF+4|Y|BR_TARGET(=BUF)|N|
|Y|N|N|PC_IF+4|N|PC_EX+4|N|
|N|Y|Y|PC_IF+4|Y|BR_TARGET|Y|
|N|Y|N|PC_IF+4|N|PC_EX+4|N|
|N|N|Y|PC_IF+4|Y|BR_TARGET|Y|
|N|N|N|PC_IF+4|N|PC_EX+4|N|

## 测试结果

### btb.S
|预测方式|预测正确次数|总预测次数|预测正确率|时钟周期数|
|:-:|:-:|:-:|:-:|:-:|
|无预测|1|101|1.0%|509|
|BTB|99|101|98.0%|311|
|BHT|99|101|98.0%|311|

### bht.S
|预测方式|预测正确次数|总预测次数|预测正确率|时钟周期数|
|:-:|:-:|:-:|:-:|:-:|
|无预测|11|110|10.0%|535|
|BTB|88|110|80.0%|379|
|BHT|97|110|88.1%|361|

### MatMul.S
|预测方式|预测正确次数|总预测次数|预测正确率|时钟周期数|
|:-:|:-:|:-:|:-:|:-:|
|无预测|274|4624|5.9%|64828|
|BTB|4076|4624|88.1%|57224|
|BHT|4346|4624|94.0%|56684|

### QuickSort.S
|预测方式|预测正确次数|总预测次数|预测正确率|时钟周期数|
|:-:|:-:|:-:|:-:|:-:|
|无预测|5334|6946|76.8%|34956|
|BTB|5114|6946|73.6%|35396|
|BHT|5838|6946|84.0%|33948|

测试中的“无预测”实际是指始终预测不跳转。

可以看到，除了快速排序样例外，BTB的预测都比不预测要好很多。在快速排序中出现了BTB弱于不预测（始终预测不跳转）的情况。推测是因为快速排序的汇编代码中采用了较多“达到结束条件后跳出循环/递归”的形式，因此始终预测不跳转的方式效果较好。而在所有样例中，BHT的预测效果都比BTB和不预测好，这也体现了BHT的强大之处。