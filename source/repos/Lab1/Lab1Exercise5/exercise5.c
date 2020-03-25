/* Lab 1 Exercise 5 program 
Modify the basic calculator so that it can read commands from a file. 
Commands are provided in the `commands.calc` file. We implement the following:
5.1 Open and closing the file in read-only mode
5.2 Modify the `readLine` function to read from a file rather than the console
We check for the `EOF` end-of-file character and return `0` when it is found.
Note that this behaviour requires any .calc files to have a blank line at the end of the file.
5.3 Modify the `main` function so that incorrect or misspelt commands cause a console error and immediate exit,
the `while` loop executes silently (no console output), and only the final total is output to console.
See http://www.cplusplus.com/reference/cstdio/ for documentation about standard input-output functions in C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Include `ctype.h` to use `isalpha` function
#include <ctype.h>

#define BUFFER_SIZE 32

/* This function takes as input a filestream `f` and a character array `buffer` to temporarily hold 
each line of characters read from `f`, and returns `0` when an `EOF` end-of-file character is read, 
or `1` otherwise. If the length of a line in the file exceeds the buffer limit, 
an error is thrown and the entire program exits early. 
We ensure the end of each line written to the `buffer` is indicated with the string termination character `\0`.  */
int readLine(FILE * f, char buffer[]) {
	int i = 0;
	char c = 0;
	while ((c = getc(f)) != '\n') {
		// Check for the `EOF` end-of-file character and return `0` when it is found.
		// Note that this behaviour requires any.calc files to have a blank line at the end of the file.
		if (c == EOF) {
			return 0;		
		}
		// Add the character `c` we read from the file `f` to the `buffer` at position `i`, then increment `i` by `1`
		buffer[i++] = c;

		/* Check to ensure that we don't write past the end of the buffer limits.
		Writing past the end of an array is called an overflow. We check the index for signs of an overflow.
		The condition below is true if we wrote to `buffer[BUFFER_SIZE - 1]` in the line above.
		This is because the final allocated index in `buffer` should be reserved for the string termination
		character `\0` in this case. The string may be terminated earlier in other cases. */
		if (i == BUFFER_SIZE) {
			// When a potential overflow is detected write an error message to `stderr` using `fprintf`
			// and then call `exit(1)` to force the program to terminate early.
			fprintf(stderr, "Error: Buffer size is too small for line input. Buffer size: %hu.\n", BUFFER_SIZE);
			exit(1);
		}
	}
	// Ensure that once the while loop has exited the buffer is correctly terminated with the string termination character
	buffer[i] = '\0';
	// Return `1` to indicate that a line was read successfully and we did not reach the end of the file
	return 1;
}

int main()
{
	// Declare `f` as a pointer to a `FILE` object and initialise it as the `NULL` pointer
	FILE* f = NULL;
	unsigned int line = 0;		// Line number counter
	float in_value, total;
	char buffer[BUFFER_SIZE];	// Variable to temporarily hold the last line read
	char command[4];
	total = 0;

	// Attempt to open the `commands.calc` file in read-only mode
	f = fopen("commands.calc", "r");
	// If we could not open the file and the pointer `f` is still `NULL` we throw an error and exit
	if (f == NULL) {
		fprintf(stderr, "Could not find the file `commands.calc`.\n");
		return;
	}

	// Reads lines from the keyboard to the `buffer` until the exit condition is satisfied
	while (readLine(f, buffer)) {
		// Increment the line number counter
		line++;
		/* Check that the line begins with three alphabetic characters followed by a space by using 
		the `isalpha` function from `ctype.h`. */
		if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ')) {
			// If the line fails the criteria, output an error "Incorrect command format" to `stderr`, then exit
			fprintf(stderr, "Incorrect command format\n");
			return;
		}

		/* Use `sscanf` to extract from the `buffer` the 3 character command and a floating point value,
		and write these to `command` and `in_value` respectively.
		See http://www.cplusplus.com/reference/cstdio/sscanf/ for documentation on `sscanf`.
		Copies data from `buffer` according to the provided `format` string to the subsequent arguments.
		The format indicates a `string` followed by a `float`, ignoring anything that comes after that (seperated by spaces).
		Thus, the initial three alphabetic characters from `buffer` will be written to `command`, then whatever comes
		after that (and before the next space or the end of the overall buffer string) will be interpreted as a `float`
		and written to the variable `in_value`. Note that this means we don't actually check for numeric characters. */
		sscanf(buffer, "%s %f", command, &in_value);

		/* Checks whether the three character `command` matches any one of "add", "sub", "mul", "div" in which case
		the `total` variable is manipulated appropriately. Otherwise, the `in_value` has still been updated. */
		if (strcmp(command, "add") == 0) {
			total += in_value;
		}
		else if (strcmp(command, "sub") == 0) {
			total -= in_value;
		}
		else if (strcmp(command, "mul") == 0) {
			total *= in_value;
		}
		else if (strcmp(command, "div") == 0) {
			total /= in_value;
		}
		/* Modify the `main` function so that incorrect or misspelt commands cause a console error and immediate exit */
		else {
			printf("Unknown command at line %u!\n", line);
			return;
		}
	}
	// Close the file
	fclose(f);
	// Output the final total to console
    printf("\tTotal is %f\n", total);
	return 0;
}
