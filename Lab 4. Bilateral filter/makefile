CFLAGS = -g -G -O3 

start: EasyBMP.o startGPU.o
	nvcc $(CFLAGS) EasyBMP.o Bilateral.o -o BF

EasyBMP.o: EasyBMP.cpp EasyBMP*.h
	nvcc $(CFLAGS) -c EasyBMP.cpp

startGPU.o: Bilateral.cu 
	nvcc -c Bilateral.cu

clean: 
	rm EasyBMP*.h
	rm EasyBMP.cpp
	rm EasyBMPtest*
	rm EasyBMPoutput*.bmp
	rm -f *.o