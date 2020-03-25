/* Lab 1 Exercise 4 program
In this program we create a rudimentary calculator which takes input from the command line. 
See http://www.cplusplus.com/reference/cstdio/ for documentation about standard input-output functions in C */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
// Include `ctype.h` to use `isalpha` function
#include <ctype.h>

#define BUFFER_SIZE 32

/* This function takes as input an array named `buffer` to hold characters, writes keyboard inputs to `buffer`,
and returns an integer `0` or `1` depending on the line of characters received from the keyboard.
First reads characters received from the keyboard and writes them to the `buffer` array until
either a `\n` keystroke is received, or the buffer limit is reached (in which case an error 
is thrown and the entire program exits early). Next ensures the end of the input string written inside the `buffer`
is indicated with the string termination character `\0`. Finally, checks if the first 4 characters
of the line received from the keyboard match "exit", in which case the integer `0` is returned,
else, the integer `0` is returned. */
int readLine(char buffer[]) {
	int i = 0;
	char c = 0;
	while ((c = getchar()) != '\n') {
		// 4.1 Complete the `while` loop by adding characters sequentially to the buffer.
		buffer[i++] = c;

		/* 4.2 Implement a check to ensure that you don't write past the end of the buffers limits.
		Writing past the end of an array is called an overflow. Check index for overflow.
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
	// 4.3 Ensure that once the while loop has exited the buffer is correctly terminated 
	// with the string termination character.
	buffer[i] = '\0';

	/* 4.4 Use the `strncmp` function to test if the line reads "exit". 
	If it does, then `readLine` should return `0`, otherwise it should return `1`. */
	// The condition is actually true (returns `0`) as long as the first 4 characters of the buffer match "exit"
	if (strncmp(buffer, "exit", 4) == 0) {
		return 0;
	}
	else {
		return 1;
	}
}

int main()
{
	float in_value, total;
	char buffer [BUFFER_SIZE];
	char command [4];
    total = 0;

	printf("Welcome to basic COM4521 calculator\nEnter command: ");

	// Reads lines from the keyboard to the `buffer` until the exit condition is satisfied
    while (readLine(buffer)) {
		/* 4.5 Check that the line begins with three alphabetic characters followed by a space. 
		You can use the `isalpha` function from `ctype.h` to check that a character is a letter. */
		if (!(isalpha(buffer[0]) && isalpha(buffer[1]) && isalpha(buffer[2]) && buffer[3] == ' ')) {
			// If the line fails the criteria, output an error "Incorrect command format" to `stderr`
			fprintf(stderr, "Incorrect command format\n");
			// Use `continue` to begin the loop again
			continue;
		}

		/* 4.6 Assuming the criteria for 4.5 is met, use `sscanf` to extract the 3 character command 
		and the floating point value from the `buffer` to `command` and `in_value` respectively.
		See http://www.cplusplus.com/reference/cstdio/sscanf/ for documentation on `sscanf`.
		Copies data from `buffer` according to the second argument `format` string to the subsequent arguments.
		The format indicates a `string` followed by a `float`, ignoring anything that comes after that (seperated by spaces).
		Thus, the initial three alphabetic characters from `buffer` will be written to `command`, then whatever comes 
		after that (and before the next space or the end of the overall buffer string) will be interpreted as a `float`
		and written to the variable `in_value`. Note that this means we don't actually check for numeric characters. */
		sscanf(buffer, "%s %f", command, &in_value);

		/* Checks whether the three character `command` matches any one of "add", "sub", "mul", "div" in which case
		the `total` variable is manipulated appropriately. Otherwise, the `in_value` has still been updated. */
		// 4.7 Change condition to check command to see if it is "add"
		if (strcmp(command, "add") == 0) { 
			total += in_value;
		}
		// 4.8 Add else if conditions for "sub", "mul", and "div"
		else if (strcmp(command, "sub") == 0) {
			total -= in_value;
		}
		else if (strcmp(command, "mul") == 0) {
			total *= in_value;
		}
		else if (strcmp(command, "div") == 0) {
			total /= in_value;
		}
		/* 4.9 Add additional conditions using `strncmp` to test the first two letters of the command. 
		If they are "ad" then output "Did you mean add?". Complete cases for "su", "mu", "di". 
		If there is a specific partial match to one of the allowed commands (i.e. the first two characters only),
		then an additional message is printed to the console. 
		There remains much left to be desired in terms of functionality and user friendliness. */
		else if (strncmp(command, "ad", 2) == 0) {
			printf("Did you mean add?\n");
			continue;
		}
		else if (strncmp(command, "su", 2) == 0) {
			printf("Did you mean sub?\n");
			continue;
		}
		else if (strncmp(command, "mu", 2) == 0) {
			printf("Did you mean mul?\n");
			continue;
		}
		else if (strncmp(command, "di", 2) == 0) {
			printf("Did you mean div?\n");
			continue;
		}
		else {
			printf("Unknown command\n");
			continue;
		}

		printf("\tTotal is %f\n", total);
		printf("Enter next command: ");
	}

    return 0;
}
