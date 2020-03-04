main: ./bin/object/ ./bin/object/main.o ./bin/object/dense_ram_node.o ./bin/object/binary.o \
			./bin/object/ram_discriminator.o
	g++ -std=gnu++14 -o ./bin/main ./bin/object/main.o ./bin/object/dense_ram_node.o \
											./bin/object/binary.o ./bin/object/ram_discriminator.o

./bin/object/:
	mkdir -p ./bin/object

./bin/object/main.o: ./tests/main.cpp ./include/ram_node.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/main.o ./tests/main.cpp

./bin/object/ram_node.o: ./lib/ram_node.cpp ./include/ram_node.hpp \
												 ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/ram_node.o ./lib/ram_node.cpp

./bin/object/dense_ram_node.o: ./lib/dense_ram_node.cpp ./include/ram_node.hpp  \
												 ./include/dense_ram_node.hpp ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/dense_ram_node.o ./lib/dense_ram_node.cpp

./bin/object/binary.o: ./lib/binary.cpp ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/binary.o ./lib/binary.cpp

./bin/object/ram_discriminator.o: ./lib/ram_discriminator.cpp \
																	./include/ram_discriminator.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/ram_discriminator.o ./lib/ram_discriminator.cpp

clean:
	rm -rf ./bin
	rm -rf ./wrapper

wrapper: ./wrapper/src ./wrapper/python
	# Generating wrapper code and python module
	swig -c++ -python -py3 -outdir ./wrapper/python -o \
	./wrapper/src/ram_node_wrap.cxx ./swig/ram_node.i

	swig -c++ -python -py3 -outdir ./wrapper/python -o \
	./wrapper/src/ram_discriminator_wrap.cxx ./swig/ram_discriminator.i
	
	# Building shared library
	python setup.py build --build-base=./wrapper/python

	# Moving shared library and cleaning up
	mv ./wrapper/python/lib.*/* ./wrapper/python/
	rm -rf ./wrapper/python/temp.* ./wrapper/python/lib.*


./wrapper/src:
	mkdir -p ./wrapper/src

./wrapper/python:
	mkdir -p ./wrapper/python