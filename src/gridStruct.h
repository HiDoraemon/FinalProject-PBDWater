#ifndef GRIDSTRUCT_H
#define GRIDSTRUCT_H

struct cell_entry{
	int begin; // index of beginning of list
	int end; // index of end of list
};

struct heap_entry{
	float r;
	int idx;
};

#endif GRIDSTRUCT_H