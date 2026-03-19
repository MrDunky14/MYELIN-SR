CXX = g++
CXXFLAGS = -std=c++17 -Wall -O3
INCLUDES = -I./include

TARGET = bin/nss_engine
SRCS = src/nss_core.cpp src/main.cpp
OBJS = $(SRCS:.cpp=.o)

all: dirs $(TARGET)

dirs:
	mkdir -p bin

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -o $(TARGET) $(OBJS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f src/*.o
	rm -rf bin

run: all
	./$(TARGET)
