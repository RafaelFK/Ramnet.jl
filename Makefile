main: ./bin/object/main.o ./bin/object/ram_node.o ./bin/object/binary.o \
			./bin/object/ram_discriminator.o
	g++ -std=gnu++14 -o ./bin/main ./bin/object/main.o ./bin/object/ram_node.o \
											./bin/object/binary.o ./bin/object/ram_discriminator.o

./bin/object/main.o: ./tests/main.cpp ./include/ram_node.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/main.o ./tests/main.cpp

./bin/object/ram_node.o: ./lib/ram_node.cpp ./include/ram_node.hpp \
												 ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/ram_node.o ./lib/ram_node.cpp

./bin/object/binary.o: ./lib/binary.cpp ./include/binary.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/binary.o ./lib/binary.cpp

./bin/object/ram_discriminator.o: ./lib/ram_discriminator.cpp \
																	./include/ram_discriminator.hpp
	g++ -std=gnu++14 -c -g -o ./bin/object/ram_discriminator.o ./lib/ram_discriminator.cpp

clean:
	rm ./bin/object/main.o ./bin/object/ram_node.o ./bin/object/binary.o ./bin/main