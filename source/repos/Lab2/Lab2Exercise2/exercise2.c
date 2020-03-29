/* Lab 2 Exercise 2 Program
This program will read in a binary file which contains 4 records of information on students.
The information consists of their forename, surname and average module mark. 
A `struct` has been defined to hold the student data, the format of this `struct` matches the
`struct` used in the program which created the binary files used in the example. 

The `student` structure uses a statically defined, fixed length `char` array to hold both the forename
and surname. This is OK but potentially wasteful when we deal with large records as much of the
`char` array will be empty. The file `students2.bin` differs from the file used in the first exercise
in that it uses dynamic length `char` arrays to hold strings. Both the forename and surname are
written to the binary file in the following format:
`unsigned int n, char[0], char[1], char[2], ..., char[n];`
The length of the `char` array is specified as an unsigned `int` at the start of each line preceding the array 
Note that as usual, the end of the character array/string (i.e. `char[n]`) is terminated with the "\0" character */

#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

/* 2. Modify the `struct` definition so that forename and surname are pointers to `char`. Now update
the code to read the student data. You will need to use `fread` to read the length of the forename (i.e. `n`). 
Hint: allocate memory for the forename (of length `n`) and then `fread` the forename, etc.
Don’t forget to also update your code to ensure that you free any memory you have allocated. */
struct student{
	char * forename;
	char * surname;
	float average_module_mark;
};

void print_student(const struct student * s);

void main(){
	struct student * students;
	int i;

	// The size of each `student` structure is now the size of two pointers (4 bytes on 32 bit machines) plus a `float` (4 bytes)
	// Whereas before the size was 2 * 128 bytes for the forename and surname plus 4 bytes for the average module mark
	// This represents a decrease in size from 260 bytes to 12 bytes (on a 32 bit machine)
	// printf("Size of `student` structure: %u\n", sizeof(struct student)); // 12 bytes = 2 * 4 + 1 * 4
	students = (struct student *) malloc(sizeof(struct student) * NUM_STUDENTS);

	// Change the name of the file we read
	FILE *f = NULL;
	f = fopen("students2.bin", "rb"); // Read-only and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find `students2.bin` file \n");
		exit(1);
	}

	// Documentation on `fread`: http://www.cplusplus.com/reference/cstdio/fread/
	for (i = 0; i < NUM_STUDENTS; i++) {
		unsigned int n;
		// Read the forename of the i-th student (first read its length `n`, allocate memory, then read)
		fread(&n, sizeof(int), 1, f);
		students[i].forename = (char *) malloc(sizeof(char) * n);
		fread(students[i].forename, sizeof(char), n, f);
		// Read the surname of the i-th student (first read its length `n`, allocate memory, then read)
		fread(&n, sizeof(int), 1, f);
		students[i].surname = (char *) malloc(sizeof(char) * n);
		fread(students[i].surname, sizeof(char), n, f);
		// Read the average module mark of the i-th student (to the memory address using `&`)
		fread(&students[i].average_module_mark, sizeof(float), 1, f);
	}
	fclose(f);

	for (i = 0; i < NUM_STUDENTS; i++){
		print_student(&students[i]);
	}
	// Don't forget to also free the data at the end of the program.
	free(students);
}

// Pointers to structures use a different member operator, arrow `->` rather than dot `.`
void print_student(const struct student * s){
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}
