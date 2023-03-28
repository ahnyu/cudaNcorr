CC := nvcc
CFLAGS := -Xcompiler -fopenmp -arch=sm_80 -lgsl -lgslcblas
CFITSIO_INC := -I/global/homes/h/hanyuz/library/local/include
CFITSIO_LIB := -L/global/homes/h/hanyuz/library/local/lib -lcfitsio
INCLUDE := -Iinclude

SRC_DIR := source
OBJ_DIR := obj
INC_DIR := include

SRCS := $(wildcard $(SRC_DIR)/*.cu)
OBJS := $(patsubst $(SRC_DIR)/%.cu, $(OBJ_DIR)/%.o, $(SRCS))

TARGET := cudaNcorr

.PHONY: all clean

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CC) $(CFLAGS) $(INCLUDE) $^ -o $@ $(CFITSIO_LIB)

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cu $(wildcard $(INC_DIR)/*.hpp)
	$(CC) $(CFLAGS) $(INCLUDE) -c $< -o $@ $(CFITSIO_INC)

clean:
	rm -f $(OBJ_DIR)/*.o $(TARGET)

