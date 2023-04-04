CC := nvcc
CFLAGS := -Xcompiler -fopenmp -arch=sm_80 -lgsl -lgslcblas
CFITSIO_INC := -I/global/homes/h/hanyuz/library/local/include
CFITSIO_LIB := -L/global/homes/h/hanyuz/library/local/lib -lcfitsio
INCLUDE := -Iinclude

SRC_DIR := source
OBJ_DIR := obj
INC_DIR := include

SRCS := $(SRC_DIR)/device.cu $(SRC_DIR)/utils.cu $(SRC_DIR)/survey.cu $(SRC_DIR)/cudaNcorr.cu $(SRC_DIR)/main.cu

OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))
DLNK := $(OBJ_DIR)/link.o

TARGET := cudaNcorr

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(DLNK) $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $^ -o $@ $(CFITSIO_LIB)

$(DLNK): $(OBJS)
	$(CC) -arch=sm_80 -dlink $^ -o $@

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(wildcard $(INC_DIR)/*.hpp)
	$(CC) $(CFLAGS) $(INCLUDE) -dc $< -o $@ $(CFITSIO_INC)

clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

