## 1.
### (1)
每4周期生成32个结果，平均每个周期生成8个结果。
实际执行过程中，平均每周期进行的浮点运算数量为：$10\times 8\times 0.85\times 80\% \times 70\% = 38.08$。
吞吐量为$38.08 \times 1.5 \times 10^9 = 57.12GFLOP/s$

### (2)
- ①：由于车道数增加1倍，因此吞吐量也增加一倍，加速比为2。
- ②：同理，其它参数不变，加速比为$\frac{15}{10} = 1.5$
- ③：同理，加速比为$\frac{0.95}{0.85} \simeq 1.12$

## 2.
### (1)
程序每一次迭代进行了6次浮点运算，进行了4次load和2次store，因此运算密度为$\frac{6}{4+2} = 1$。

### (2)
不妨设单精度数占4字节。汇编代码如下：
```
        ADDI    R2, Ra_re, #1200        ; a的实部数组末尾地址
        ADDI    R1, R0, #44             ; 第一段长度
        MOVI2S  VLR, R1
        ADDI    R1, R0, #176            ; 第一段的字节数
        ADDI    R3, R0, #64             ; 后面每一段的长度
Loop:
        LV      V1, Ra_re
        LV      V2, Rb_re
        MULTV   V5, V1, V2
        LV      V3, Ra_im
        LV      V4, Rb_im               
        MULTV   V6, V3, V4
        SUBV    V5, V5, V6
        SV      Rc_re, V5
        MULTV   V5, V1, V4
        MULTV   V6, V2, V3
        ADDV    V5, V5, V6
        SV      Rc_im, V5
        ADD     Ra_re, Ra_re, R1
        ADD     Rb_re, Rb_re, R1
        ADD     Ra_im, Ra_im, R1
        ADD     Rb_im, Rb_im, R1
        ADDI    R1, R0, #256            ; 下一段的字节数
        MOVI2S  VLR, R3                 ; 将向量长度设为64
        SUB     R4, R2, Ra_re           ; 是否到达末尾
        BNEZ    R4, Loop                ; 否则继续循环

```
其中，Ra_re, Rb_re等是存放复数向量的实部/虚部元素地址的寄存器
### (3)
可以略微调整循环语句顺序，在循环开始前load a和b向量当前段的实部，在循环中计算c的虚部时load a和b向量下一段的实部：
```
chime1: MULTV   V5, V1, V2      ; 这里a和b的实部已在上一次迭代时完成载入
        LV      V3, Ra_im       ; 启动开销15周期
chime2: LV      V4, Rb_im               
        MULTV   V6, V3, V4      ; 启动开销23周期
chime3: SUBV    V5, V5, V6
        SV      Rc_re, V5       ; 启动开销20周期
chime4: MULTV   V5, V1, V4
        LV      V1, Ra_re       ; 这里load下一段的V1，启动开销23周期（这里实际有数据相关，在乘法计算完a_re*b_im后，
                                ; 才能load下一个a_re）
chime5: MULTV   V6, V2, V3     
        LV      V2, Rb_re       ; 这里load下一段的V2，启动开销23周期
chime6: ADDV    V5, V5, V6
        SV      Rc_im, V5       ; 启动开销20周期
```
这里有6个chime，由于一次迭代中有6个访存指令，而只有一个存储器流水线，因此至少有6个chime。这一结果已是最优。
根据Strip Mining的计算式：
$T_n = \lceil \frac{n}{MVL} \rceil \times(T_{loop} + T_{start}) + n \times T_{chime}$
这里题目中未给出串行指令的执行时间，因此忽略$T_{loop}$，那么计算300个值的总开销为：
$5 \times (15+23+20+23+23+20) + 300 \times 6 = 2420$
如果将一组实部和虚部看作一个复数值，则平均每个复数值的开销为$\frac{2420}{300} \simeq 8.07$周期
如果将实部或虚部看作一个复数值（即认为计算了600个值），那么平均每个值的开销为上面的一半。

### (4)
这里假定向量乘法和向量加减采用的功能部件是相互独立的（即向量乘法和向量加减可以同时进行），且都只有一个。
```
chime1: LV      V1, Ra_re
        LV      V2, Rb_re
        MULTV   V5, V1, V2
        LV      V3, Ra_im       ; 启动开销为23周期，这里MULTV和上一条LV间有相关
chime2: LV      V4, Rb_im               
        MULTV   V6, V3, V4
        SUBV    V5, V5, V6
        SV      Rc_re, V5       ; 启动开销为43周期，这里每一条指令间都有相关
chime3: MULTV   V5, V1, V4      ; 启动开销为8周期
chime4: MULTV   V6, V2, V3      
        ADDV    V5, V5, V6
        SV      Rc_im, V5       ; 启动开销为28周期
```
这里有4个chime。由于有4条MULTV指令，因此至少有4个chime。这一结果已是最优。
总的开销为$5\times(23+43+8+28)+300\times4 = 1710$周期，平均每个复数值的开销为$\frac{1710}{300} = 5.7$周期。