CC = gcc
AR = ar rcv
RANLIB = ranlib 
CFLAGS = -O3 -D_LARGEFILE64_SOURCE=1 -fPIC

OBJS = adler32.o compress.o crc32.o deflate.o gzclose.o gzlib.o gzread.o \
	gzwrite.o infback.o inffast.o inflate.o inftrees.o trees.o uncompr.o zutil.o
HEADERS = crc32.h deflate.h gzguts.h inffast.h inffixed.h inflate.h \
		  inftrees.h trees.h zconf.h zlib.h zutil.h
FILES = adler32.c compress.c crc32.c deflate.c example.c gzclose.c \
		gzlib.c gzread.c gzwrite.c infback.c inffast.c inflate.c inftrees.c\
		minigzip.c trees.c uncompr.c zutil.c

libz.a: $(OBJS) $(HEADERS) $(FILES)
	$(AR) libz.a $(OBJS)
	$(RANLIB) libz.a

rofu:
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o adler32.o adler32.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o compress.o compress.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o crc32.o crc32.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o deflate.o deflate.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o gzclose.o gzclose.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o gzlib.o gzlib.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o gzread.o gzread.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o gzwrite.o gzwrite.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o infback.o infback.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o inffast.o inffast.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o inflate.o inflate.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o inftrees.o inftrees.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o trees.o trees.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o uncompr.o uncompr.c
	gcc -O3 -D_LARGEFILE64_SOURCE=1 -c -o zutil.o zutil.c
	ar rc libz.a adler32.o compress.o crc32.o deflate.o gzclose.o gzlib.o gzread.o gzwrite.o infback.o inffast.o inflate.o inftrees.o trees.o uncompr.o zutil.o 


.o:
	$(CC) $(CFLAGS) -c -o $*.o $*.c

clean:
	rm -rf *.o *.a *~
