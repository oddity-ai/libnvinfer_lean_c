CXX        := g++
CXXFLAGS   := -O3 -Wall -Wextra -pedantic -fPIC
LDFLAGS    := -shared

TARGET     := libnvinfer_lean_c.so
STATIC_LIB := libnvinfer_lean_static.a
SRC        := nvinfer_lean_c.cpp
OBJ        := $(SRC:.cpp=.o)

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ) $(STATIC_LIB)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) -Wl,--whole-archive $(STATIC_LIB) -Wl,--no-whole-archive

clean:
	$(RM) $(OBJ) $(TARGET)

.PHONY: all clean
