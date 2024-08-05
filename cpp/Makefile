run:
	@nvcc -c -o mm.o mm.cu
	@g++ -c -o custom.o custom.cpp
	@nvcc -o main.o custom.o mm.o
	@./main.o