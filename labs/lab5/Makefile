# 指定编译器
CXX = g++
CUDA = nvcc
FLAG = -O2 -mfma
MACRO = "-DN=(1 << 10)" -DBLOCK_LEN=16 -DVERIFY

# 指定源文件目录和目标文件目录
CPP_SRC_DIR = ./MatMul
CUDA_SRC_DIR = ./MatMul_CUDA
CPP_OBJ_DIR = ./build/MatMul
CUDA_OBJ_DIR = ./build/MatMul_CUDA

# 找到所有的 .cpp 文件
CPP_SRCS = $(wildcard $(CPP_SRC_DIR)/*.cpp)

# 找到所有的 .cu文件
CUDA_SRCS = $(wildcard $(CUDA_SRC_DIR)/*.cu)

# 将 .cpp 文件列表转换为 .o 文件列表
CPP_OBJS = $(patsubst $(CPP_SRC_DIR)/%.cpp,$(CPP_OBJ_DIR)/%.o,$(CPP_SRCS))

# 将 .cu 文件列表转换为 .o 文件列表
CUDA_OBJS = $(patsubst $(CUDA_SRC_DIR)/%.cu,$(CUDA_OBJ_DIR)/%.o,$(CUDA_SRCS))

# 默认目标
all: cpp cuda

cpp: $(CPP_OBJS)

# 编译规则
$(CPP_OBJ_DIR)/%.o: $(CPP_SRC_DIR)/%.cpp
	@mkdir -p $(@D)
	@$(CXX) $(FLAG) $(MACRO) $< -o $@

cuda: $(CUDA_OBJS)
$(CUDA_OBJ_DIR)/%.o: $(CUDA_SRC_DIR)/%.cu
	@mkdir -p $(@D)
	@$(CUDA) -O2 $(MACRO) $< -o $@

# 清理目标
clean:
	@rm -rf build

.PHONY: all cpp cuda clean