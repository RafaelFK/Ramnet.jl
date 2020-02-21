main: ./bin/object/main.o ./bin/object/ram_node.o ./bin/object/binary.o
	g++ -std=gnu++14 -o ./bin/main ./bin/object/main.o ./bin/object/ram_node.o \
											./bin/object/binary.o

./bin/object/main.o: ./tests/main.cpp ./include/ram_node.hpp
	g++ -std=gnu++14 -c -o ./bin/object/main.o ./tests/main.cpp

./bin/object/ram_node.o: ./lib/ram_node.cpp ./include/ram_node.hpp \
												 ./include/binary.hpp
	g++ -std=gnu++14 -c -o ./bin/object/ram_node.o ./lib/ram_node.cpp

./bin/object/binary.o: ./lib/binary.cpp ./include/binary.hpp
	g++ -std=gnu++14 -c -o ./bin/object/binary.o ./lib/binary.cpp

clean:
	rm ./bin/object/main.o ./bin/object/ram_node.o ./bin/object/binary.o ./bin/main