
slic:
	nvcc -D DEBUG -O3 -arch=sm_20 slic.cu cimage/imagebmp.c converter.cu -o slic


clean:
	-rm slic
	-rm output_bmp/iter*.bmp
	-rm output_bmp/iter*.BMP


