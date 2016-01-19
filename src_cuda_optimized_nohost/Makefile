CC = g++
LD = g++

WARNGCC= -Wno-sign-compare -Wno-reorder -Wno-unknown-pragmas -Wno-overloaded-virtual

# --- With optimisation
CPPFLAGS = -std=c++0x -DNDEBUG -DSHUT_UP -O3 -msse2 -Wall $(WARNGCC)
LDFLAGS = -DNDEBUG -O3

# --- Debugging
#CPPFLAGS = -std=c++0x -g -Wall $(WARNGCC) 
#LDFLAGS = 

# --- CUDA
CFLAGS=$(CCONFIG)
CFLAGS+=`pkg-config --cflags opencv` -O3 -DSHUT_UP -DNDEBUG
LFLAGS=$(LCONFIG) `pkg-config opencv --libs` -lstdc++ -DSHUT_UP -DNDEBUG


INCLUDE_DIR =
LIB_DIR =
LIBS = (LCONFIG) `pkg-config opencv --libs` -lstdc++

simple:	sf1_gpu


testgpu:
	./sf1_gpu simple-data/config.txt 6 simple-data/tree

%.o: %.cpp 
	$(CC) -c $(CPPFLAGS) $(INCLUDE_DIR) $< -o $@

%.o: %.cu 
	nvcc  $(CFLAGS) -cudart static --machine 64 --compile -rdc=true $< -o $@

main_test_simple.o: main_test_simple.cpp
	nvcc  $(CFLAGS) -cudart static --machine 64 --compile -rdc=true -o $@ $<

sf1_gpu: ConfigReader.o ImageData.o ImageDataFloat.o labelfeature.o label.o main_test_simple.o NoPointerFunctions.o kernel.o
	nvcc $+ -o $@ $(LDFLAGS) $(LFLAGS)

clean:
	rm -f *.o sf1_gpu

