.PHONY:	all clean

all:
	./setup.sh make -f Makefile2 $(MAKECMDGOALS) -j 8

clean:
	./setup.sh make -f Makefile2 clean
