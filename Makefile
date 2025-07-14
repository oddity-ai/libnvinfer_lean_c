CXX        := g++
CXXFLAGS   := -O3 -Wall -Wextra -pedantic -fPIC

LDFLAGS    := -shared
LIBS       := -lnvinfer_lean

TARGET     := libnvinfer_lean_c.so
SRC        := nvinfer_lean_c.cpp
OBJ        := $(SRC:.cpp=.o)

all: $(TARGET)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJ)
	$(CXX) $(LDFLAGS) -o $@ $(OBJ) $(LIBS)

clean:
	$(RM) $(OBJ) $(TARGET)

.PHONY: all clean
