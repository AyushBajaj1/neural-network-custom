mnist: src/Matrix.cpp src/Network.cpp entry/mnist.cpp src/Matrix.h src/Network.h 
	clang++ -std=c++20 -Wall -Wextra -Wpedantic -O3 src/Matrix.cpp src/Network.cpp entry/mnist.cpp -o mnist

london: src/Matrix.cpp src/Network.cpp src/Normalizer.cpp entry/london_housing.cpp src/Matrix.h src/Network.h src/Normalizer.h
	clang++ -std=c++20 -Wall -Wextra -Wpedantic -O3 src/Matrix.cpp src/Network.cpp src/Normalizer.cpp entry/london_housing.cpp -o london

custom: src/Matrix.cpp src/Network.cpp entry/custom_data.cpp src/Matrix.h src/Network.h
	clang++ -std=c++20 -Wall -Wextra -Wpedantic -O3 src/Matrix.cpp src/Network.cpp entry/custom_data.cpp -o custom

clean:
	rm -f mnist
	rm -f london
	rm -f custom