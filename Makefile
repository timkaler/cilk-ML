.PHONY:	all clean

all:
	./setup.sh make VERBOSE=1 -f Makefile2 $(MAKECMDGOALS) -j 1

serial:
	./setup.sh make VERBOSE=1 -f Makefile2_serial -j 8

plocks:
	./setup.sh make VERBOSE=1 -f Makefile2_PLOCKS -j 8

semisort:
	./setup.sh make -f Makefile2 build/semisort -j 8

clean:
	./setup.sh make -f Makefile2 clean
