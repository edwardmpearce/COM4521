/* Lab 2 Exercise 3 Program
This program will read in a binary file which contains records of information on students.
The previous two exercises assumed we knew how many student records were stored in the binary data file.
Now we will update our program to read, store and display an arbitrary number of records.
In order to do this we are going to use a linked list data structure.
The `linked_list.h` header file contains very basic implementation of a generic linked list. The header
file contains a structure `llitems` which defines a pointer to the previous and next item in the list.
3.1 The implementation of a linked list is incomplete.
Complete the function `add_to_linked_list` by implementing the following:
3.1.1 Check that the `ll_end` item is in fact the end of the list (the next record should be `NULL`).
If it is not the end then the function should return `NULL`.
3.1.2 Create and add a new item to the end of the passed linked list,
updating the old end of the linked list to reflect the addition.
3.1.3 Return a pointer to the new end of the linked list. */

/* Define a structure called `llitem` (linked list item) which (recursively) contains pointers `previous` and `next` to other
`llitem` structures as well as a generic `void` pointer called `record` which can point to a data record, then
use the `typedef` keyword to create an alias to this newly defined structure called `llitem`, so we can omit writing `struct` 
This `struct` has size equal to 3 pointers (each 4 bytes on a 32 bit machine), so 12 bytes in total */
typedef struct llitem{
	struct llitem * previous;
	struct llitem * next;
	void * record;
} llitem;

/* Declaring `print_callback` as a pointer to a function which takes as input a `void` pointer `r` and 
outputs a `void` pointer, then defining the function pointer as the `NULL` pointer */
void (*print_callback)(void* r) = NULL;

/* Defining `print_items` as a function which takes as input a pointer to an `llitem` structure called `ll_start` 
and outputs a pointer of `void` type */
void print_items(llitem * ll_start){
	// Define `ll` as pointer to an `llitem` structure and assign to it the same pointer value as `ll_start`
	// i.e. initialise `ll` to point to the start of our linked list
	llitem * ll = ll_start;
	// While `ll` is not the `NULL` pointer, i.e. while we havent reached the end of the linked list
	while (ll != NULL){
		// If the `print_callback` function pointer has been assigned to a non-`NULL` value
		// We call the corresponding function on the `record` pointer of the linked list `ll` at the current position
		if (print_callback != NULL)
			print_callback(ll->record);
		// Move the `ll` pointer to the next address in the linked list
		ll = ll->next;
	}
}

/* Creates a new linked list with one (empty) item by allocating memory and returning a pointer */
llitem * create_linked_list(){
	// Declare `ll_start` to be a pointer to an `llitem` structure. This will be our output.
	llitem * ll_start;
	// Define `ll_start` by allocating an appropriate amount of memory and explicitly casting the resulting pointer correctly
	ll_start = (llitem*)malloc(sizeof(llitem));
	// Initialise the `next`, `previous`, and `record` pointers as `NULL`
	ll_start->next = NULL;
	ll_start->previous = NULL;
	ll_start->record = NULL;
	return ll_start;
}

/* Extends the passed linked list by adding a new (empty) item to the end. 
Input: pointer to the end of the linked list to be extended. Output: pointer to the end of the extended list */
llitem * add_to_linked_list(llitem * ll_end) {
	// Check to make sure that the pointer provided points to an `llitem` structure which has been allocated
	// Check that the `ll_end` item is in fact the end of the list (the next record should be `NULL`).
	if (ll_end == NULL || ll_end->next != NULL) {
		// Otherwise, the function will return `NULL`
		return NULL;
	}
	/* Create a new `llitem` structure pointer, `ll`, to be added to the end of the passed linked list, 
	allocating memory and initialising the `next`, `previous`, and `record` pointers appropriately */
	llitem* ll = (llitem*)malloc(sizeof(llitem));
	// Update the new end of the linked list to point backwards to the old end
	ll->previous = ll_end;
	// Set the `next` pointer of the new end of the linked list to point to `NULL`
	ll->next = NULL;
	// Initialise the `record` pointer as `NULL`
	ll->record = NULL;
	// Add the new item to the end of the passed linked list by updating the old end of the linked list to reflect the addition
	ll_end->next = ll;
	// Return a pointer to the new end of the linked list
	return ll;
}

// Sequentially free the pointers in the linked list whose starting address is `ll_start`
// Note that this doesn't free the records which the linked list points to, so these should be freed separately
void free_linked_list(llitem *ll_start){
	llitem *ll = ll_start;
	while (ll != NULL){
		llitem *temp = ll->next;
		free(ll);
		ll = temp;
	}
}
