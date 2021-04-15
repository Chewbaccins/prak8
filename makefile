all: reflection.cpp
	mpicxx -o "ref" "reflection.cpp"

convert: convert.cpp
	g++ -o "convert" convert.cpp

polus: reflection.cpp
	mpicxx -o "ref" "reflection.cpp" -std=c++11

queue: polus
	bsub -n $(NP) -W 00:12 -o result.%J.out ./ref $(SIZE)