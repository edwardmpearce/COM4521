#define ESCAPE_RADIUS_SQ 2000.0*2000.0	// The escape radius squared

// Declare an `enum` data type for transfer function types
/* See the following links for more details on the syntax and usage of `enum` data types
https://www.geeksforgeeks.org/enumeration-enum-c/
https://www.programiz.com/c-programming/c-enumeration
https://en.cppreference.com/w/c/language/enum 
Then use `typedef` to create an alias so we can omit the `enum` keyword in subsequent definitions */
typedef enum TRANSFER_FUNCTION{
	ESCAPE_VELOCITY,
	HISTOGRAM_ESCAPE_VELOCITY,
	HISTOGRAM_NORMALISED_ITERATION_COUNT,
	RANDOM_NORMALISED_ITERATION_COUNT
} TRANSFER_FUNCTION;

// Declare an `enum` data type for methods to increment the histogram counters whilst avoiding race conditions
// Then use `typedef` to create an alias so we can omit the `enum` keyword in subsequent definitions
typedef enum HISTOGRAM_METHOD {
	SERIAL,
	CRITICAL_SECTION,
	LOCAL_HIST_AND_COMBINE,
	OMP_ATOMIC
} HISTOGRAM_METHOD;

// Declare structure `rgb` which holds three values between 0-255
// Then use `typedef` to create an alias so we can omit the `struct` keyword in subsequent usage
typedef struct rgb{
	unsigned char r;
	unsigned char g;
	unsigned char b;
} rgb;

// Declarations for transfer functions, which take `int` pixel coordinates (x,y) as input and return an `rgb` structure
// Escape velocity transfer function
rgb ev_transfer(int x, int y);
// Histogram escape velocity transfer with equally distributed colours regardless of iterations
rgb h_ev_transfer(int x, int y);
// Histogram normalised iteration count (NIC) transfer with smooth shading (no banding) and equally distributed colours regardless of iterations
rgb h_nic_transfer(int x, int y);
// Random colours normalised iteration count (NIC) transfer with smooth shading (no banding)
rgb rand_nic_transfer(int x, int y);
