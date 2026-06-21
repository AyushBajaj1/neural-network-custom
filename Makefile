TARGET = main

CXX = clang++

CXXFLAGS = -std=c++20 -g -Wall -Wextra -O2 -MMD -MP -fsanitize=address,undefined

SRCDIR = src
BUILDDIR = build

SRCS = $(SRCDIR)/main.cpp
OBJS = $(SRCS:$(SRCDIR)/%.cpp=$(BUILDDIR)/%.o)
DEPS = $(OBJS:.o=.d)

all: $(TARGET)

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) $(OBJS) -o $(TARGET)

-include $(DEPS)

$(BUILDDIR)/%.o: $(SRCDIR)/%.cpp | $(BUILDDIR)
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILDDIR):
	mkdir -p $(BUILDDIR)

clean:
	rm -f $(OBJS) $(TARGET) $(DEPS)

.PHONY: all clean