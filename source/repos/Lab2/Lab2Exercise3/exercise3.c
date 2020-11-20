/* Lab 2 Exercise 3 Program
This program will read in a binary file which contains records of information on students.
The previous two exercises assumed we knew how many student records were stored in the binary data file. 
Now we will update our program to read, store and display an arbitrary number of records. 
In order to do this we are going to use a linked list data structure. 
The `linked_list.h` header file contains very basic implementation of a generic linked list. The header
file contains a structure `llitems` which defines a pointer to the previous and next item in the list. */
#include <stdio.h>
#include <stdlib.h>
#include "linked_list.h"
/* 3.2 In order to use the `print_items` function (defined in `linked_list.h`), the function pointer `print_callback`
(declared in `linked_list.h`) must be set to a function with a `void` type pointer as input and output.
Such a `print_function` would have the following declaration structure:
`void print_function(void *);`
We already have the `print_student` function but this function accepts a `const` pointer to `student` structure as input. 
Assign the `print_callback` pointer to the `print_student` function modified with an explicit cast of the input. 
You must be careful about your use of brackets here. */

#define NUM_STUDENTS 4

struct student {
	char* forename;
	char* surname;
	float average_module_mark;
};

void print_student(const struct student* s);

/* 3.3 Update your code to read in `students2.bin` by creating a linked list of `student` records. 
You will need a pointer to mark both the start and end of the linked list. 
To test if your stream is at the end of a file (i.e. it has read the last record) you should check 
the return value of `fread` (if less than the requested number of items are returned this indicates the end of the file). 
You should use the `create_linked_list` and `add_to_linked_list` functions. 
You can use the `free_linked_list` function to free your linked list but be careful 
as this won't free the records which the linked list points to. */
void main() {
	llitem* start = NULL;
	llitem* end = NULL;
	unsigned int str_len;	// To (temporarily) store the length of passed strings (for `forename` and `surname`)

	// Set the generic linked list `print_callback` function pointer to point to our application's `print_student` function 
	// Using explicit casting to bring the `&print_student` function pointer into the required format (a function receiving and outputing `void` pointers) 
	print_callback = (void (*)(void*)) &print_student;

	FILE* f = NULL;
	f = fopen("students2.bin", "rb"); // Read-only and binary flags
	if (f == NULL) {
		fprintf(stderr, "Error: Could not find `students2.bin` file \n");
		exit(1);
	}

	// Read the student data from the file into a linked list
	// Documentation on `fread`: http://www.cplusplus.com/reference/cstdio/fread/
	/* We attempt to read 1 `unsigned int` (4 bytes) from the start of each record and store it in the `str_len` variable
	This `unsigned int` should represent the length of the student's forename in the record
	If `fread` doesn't return `1` to denote successfully reading 1 `unsigned int`, then 
	there are no more records to read and we break from the loop */
	while (fread(&str_len, sizeof(unsigned int), 1, f) == 1) {
		// Allocate memory and create a pointer to a `student` structure to hold the record from file
		// A new memory address will be allocated to `s` on each iteration, which we later store in our linked list
		struct student* s;
		s = (struct student*)malloc(sizeof(struct student));

		// Allocate memory, then read the forename of the current student (forename length has already been stored in `str_len`)
		s->forename = (char*)malloc(sizeof(char) * str_len);
		fread(s->forename, sizeof(char), str_len, f);

		// Read the surname of the current student (first read its length, then allocate memory and read)
		fread(&str_len, sizeof(unsigned int), 1, f);
		s->surname = (char*)malloc(sizeof(char) * str_len);
		fread(s->surname, sizeof(char), str_len, f);

		// Read the average module mark of the current student (to the memory address using `&`)
		fread(&s->average_module_mark, sizeof(float), 1, f);

		// Append a new item to the linked list
		if (start == NULL) {
			// Initial case
			start = create_linked_list();
			end = start;
		}
		else {
			// General case
			end = add_to_linked_list(end);
		}
		// Set the `record` pointer at the `end` of the list (cast the `student` structure pointer `s` as a generic `void` pointer)
		end->record = (void*)s;
	}
	fclose(f);

	// Print the items in our linked list according to the function assigned to `print_callback` (i.e. the `print_student` function)
	print_items(start);

	// Cleanup the records by freeing the allocated memory
	llitem* current_position = start;
	while (current_position != NULL) {
		free(((struct student*)current_position->record)->forename);
		free(((struct student*)current_position->record)->surname);
		free((struct student*)current_position->record);
		current_position = current_position->next;
	}
	free_linked_list(start);
}

// Pointers to structures use a different member operator, arrow `->` rather than dot `.`
void print_student(const struct student* s) {
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}
