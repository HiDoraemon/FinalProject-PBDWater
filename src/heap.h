// Max Heap Functions

#ifndef HEAP_H
#define HEAP_H

// Get the max of the passed in heap
__host__ __device__ float heapGetMax(heap_entry* heap){
	if(!heap){
		return -1; // error checking
	}
	return heap[0].r;
}

// Heapify for max heap
__host__ __device__ void heapifyDown(heap_entry* heap, int size){
	int entry = 0, left = 1, right = 2, new_entry = -1;
	float entry_val = heap[entry].r, left_val = heap[left].r, right_val = heap[right].r;
	heap_entry temp;

	do{
		if(left <= size && left_val > entry_val){
			new_entry = left;
		}
		if(right <= size && right_val > entry_val){
			new_entry = right;
		}
		if(entry != new_entry){
			temp = heap[entry];
			heap[entry] = heap[new_entry];
			heap[new_entry] = temp;
			entry = new_entry;
			left = 2 * entry + 1;
			right = 2 * (entry + 1);
			entry_val = heap[entry].r;
			left_val = heap[left].r;
			right_val = heap[right].r;
		}
	}while((right_val > entry_val || left_val > entry_val) && entry != new_entry);
}

// Removes from heap
__host__ __device__ void heapRemoveMax(heap_entry* heap, int& size){
	heap[0] = heap[size - 1];
	--size;
	heapifyDown(heap, size);
}

// Up heap
__host__ __device__ void heapifyUp(heap_entry* heap, int size){
	int parent = (size - 1) / 2, entry = size - 1;
	float parent_val = heap[parent].r, entry_val = heap[entry].r;

	while(parent_val < entry_val){
		// swap
		heap_entry temp = heap[parent];
		heap[parent] = heap[entry];
		heap[entry] = temp;

		// change index
		entry = parent;
		entry_val = parent_val;
		parent = entry / 2;
		parent_val = heap[parent].r;
	}
}

// Inserts into the heap
__host__ __device__ void heapInsert(heap_entry* heap, float value, int i, int& size, int max_size){
	heap_entry entry;
	entry.idx = i;
	entry.r = value;
	heap[size] = entry;
	heapifyUp(heap, size);
	++size;
	size = min(max_size, size);
}

#endif HEAP_H