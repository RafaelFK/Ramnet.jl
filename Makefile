current_dir = $(shell pwd)
include_path = $(current_dir)/include

main: ./bin/object/ ./bin/object/main.o ./bin/object/dense_ram_node.o ./bin/object/sparse_ram_node.o ./bin/object/binary.o \
			./bin/object/ram_discriminator.o
	g++ -std=gnu++14 -o ./bin/main -I$(include_path) ./bin/object/main.o ./bin/object/dense_ram_node.o ./bin/object/sparse_ram_node.o \
											./bin/object/binary.o ./bin/object/ram_discriminator.o

./bin/object/:
	mkdir -p ./bin/object

./bin/object/main.o: ./tests/main.cpp ./include/ram_node.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/main.o -I$(include_path) ./tests/main.cpp

./bin/object/ram_node.o: ./lib/ram_node.cpp ./include/ram_node.hpp \
												 ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/ram_node.o -I$(include_path) ./lib/ram_node.cpp

./bin/object/dense_ram_node.o: ./lib/dense_ram_node.cpp ./include/node.hpp  \
												 ./include/dense_ram_node.hpp ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/dense_ram_node.o -I$(include_path) ./lib/dense_ram_node.cpp

./bin/object/sparse_ram_node.o: ./lib/sparse_ram_node.cpp ./include/node.hpp \
																./include/sparse_ram_node.hpp ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/sparse_ram_node.o -I$(include_path) ./lib/sparse_ram_node.cpp

./bin/object/binary.o: ./lib/binary.cpp ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/binary.o -I$(include_path) ./lib/binary.cpp

./bin/object/ram_discriminator.o: ./lib/ram_discriminator.cpp \
																	./include/ram_discriminator.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/ram_discriminator.o  -I$(include_path) ./lib/ram_discriminator.cpp

clean:
	rm -rf ./bin
	rm -rf ./wrapper/python/ramnet/
	rm -rf ./wrapper/src/

wrapper: ./wrapper/src ./wrapper/python/ramnet/node ./wrapper/python/ramnet/model
	# Generating wrapper code and python module

	swig -c++ -python -py3 -outdir ./wrapper/python/ramnet/model -o \
	./wrapper/src/ram_discriminator_wrap.cxx ./swig/ram_discriminator.i
	
	swig -c++ -python -py3 -outdir ./wrapper/python/ramnet/node -o \
	./wrapper/src/dense_ram_node_wrap.cxx ./swig/dense_ram_node.i

	swig -c++ -python -py3 -outdir ./wrapper/python/ramnet/node -o \
	./wrapper/src/sparse_ram_node_wrap.cxx ./swig/sparse_ram_node.i

	# Building shared library
	cd ./wrapper/python; python setup.py build --build-base=$(current_dir)/wrapper/python

	# Moving shared library and cleaning up
	mv ./wrapper/python/lib.*/*node* ./wrapper/python/ramnet/node
	mv ./wrapper/python/lib.*/*discriminator* ./wrapper/python/ramnet/model
	rm -rf ./wrapper/python/temp.* ./wrapper/python/lib.*


./wrapper/src:
	mkdir -p ./wrapper/src

./wrapper/python/ramnet/node:
	mkdir -p ./wrapper/python/ramnet/node
	touch ./wrapper/python/ramnet/__init__.py
	touch ./wrapper/python/ramnet/node/__init__.py

./wrapper/python/ramnet/model:
	mkdir -p ./wrapper/python/ramnet/model
	touch ./wrapper/python/ramnet/model/__init__.py