default_target: all

all:clean init compile build run

run:
	./build/bin/opengl-research

build:
	cd build;make

compile:
	cd build;cmake ..

init:
	mkdir -p build

clean:
	rm -rf build

.PHONY : all clean init compile build run all

