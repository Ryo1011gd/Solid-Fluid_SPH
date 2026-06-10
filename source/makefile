TARGET = MK-SPH

CC = /opt/nvidia/hpc_sdk/Linux_x86_64/25.11/compilers/bin/nvc++
OBJE = main_Implicit.o errorfunc.o log.o

CFLAGS = -acc -O3 -Minfo=accel -gpu=cc120 -cuda
LDFLAGS = -lm
WFLAGS =

$(TARGET): $(OBJE)
	$(CC) $(CFLAGS) $(WFLAGS) -o $@ $(OBJE) $(LDFLAGS)

.cpp.o:
	$(CC) $(CFLAGS) $(WFLAGS) -c $<

clean:
	/bin/rm -f *.o *~ $(TARGET) $(TARGET).exe

main_Implicit.o		: log.h
errorfunc.o		: errorfunc.h
log.o			: errorfunc.h log.h
