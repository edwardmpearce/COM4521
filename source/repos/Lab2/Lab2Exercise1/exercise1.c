/* Lab 2 Exercise 1 Program
The purpose of this exercise is to modify some existing code to use pointers. 
The example code will read in a binary file which contains 4 records of information on students.
The information consists of their forename, surname and average module mark. 
A `struct` has been defined to hold the student data, the format of this `struct` matches the
`struct` used in the program which created the binary files used in the example. 
1.1 Compile and execute the program. It should print out the information for 4 students. */

#include <stdio.h>
#include <stdlib.h>

#define NUM_STUDENTS 4

struct student{
	char forename[128];
	char surname[128];
	float average_module_mark;
};

/* 1.2 The `print_student` function is inefficient. It requires passing a structure (by value) which
causes all of the data to be duplicated. Amend this so that the structure is passed as a reference.
You will need to update both the `print_student` function declaration and definition. */
// Function declaration
void print_student(const struct student* s);

// Function definition
// Pointers to structures use a different member operator, arrow `->` rather than dot `.`
void print_student(const struct student* s) {
	printf("Student:\n");
	printf("\tForename: %s\n", s->forename);
	printf("\tSurname: %s\n", s->surname);
	printf("\tAverage Module Mark: %.2f\n", s->average_module_mark);
}

/* 1.3 The `main` function uses a statically defined array to hold our student data. 
Modify this code so that `students` is a pointer to a student `struct`
and then manually allocate enough memory to read in the student records. 
Don't forget to also free the data at the end of the program. */
void main(){
	struct student * students;
	int i;

	// Allocate a piece of memory equal to the size of the `student` structure multiplied by the number of students
	// Then explicitly cast the pointer to that memory as a pointer to a `student` structure, and assign to the variable `students`
	// printf("Size of `student` structure: %u\n", sizeof(struct student)); // 260 bytes = 2 * 128 * 1 + 1 * 4
	students = (struct student *) malloc(sizeof(struct student) * NUM_STUDENTS);

	FILE *f = NULL;
	f = fopen("students.bin", "rb"); // Read-only and binary flags
	if (f == NULL){
		fprintf(stderr, "Error: Could not find `students.bin` file \n");
		exit(1);
	}

	// Read the data from the file `f` into the `students`
	// Documentation on `fread`: http://www.cplusplus.com/reference/cstdio/fread/
	fread(students, sizeof(struct student), NUM_STUDENTS, f);
	fclose(f);

	for (i = 0; i < NUM_STUDENTS; i++){
		print_student(&students[i]);
	}
	// Don't forget to also free the data at the end of the program.
	free(students);
}
