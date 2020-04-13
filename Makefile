.PHONY:	all clean

all:
	mkdir -p build
	./setup.sh make VERBOSE=1 -f Makefile2 -j 8

comparisons: all wl serial plocks

wl:
	mkdir -p build_wl
	./setup.sh make VERBOSE=1 -f Makefile2_wl -j 8

serial:
	mkdir -p build_serial
	./setup.sh make VERBOSE=1 -f Makefile2_serial -j 8

plocks:
	mkdir -p build_plocks
	./setup.sh make VERBOSE=1 -f Makefile2_PLOCKS -j 8

semisort:
	./setup.sh make -f Makefile2 build/semisort -j 8

clean:
	./setup.sh make -f Makefile2 clean
	./setup.sh make -f Makefile2_wl clean
	./setup.sh make -f Makefile2_PLOCKS clean
	./setup.sh make -f Makefile2_serial clean
	rm -rf build build_wl build_serial build_plocks
