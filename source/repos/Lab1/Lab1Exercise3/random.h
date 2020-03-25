#define RAND_SEED 123

/* 2.2 Modify `random.h` by adding appropriate function declarations. If you don’t include the
`extern` keyword then it will be implicitly defined by the compiler (as all globals are `extern` by
default). It is good practice to include it. The project should now build without errors. */
extern void init_random();
extern unsigned short random_ushort();

/* 2.3 We are now going to implement simple linear congruential generator 
See https://en.wikipedia.org/wiki/Linear_congruential_generator for details. 
This works by advancing a seed by a simple multiplication and addition. 
Define the parameters `RANDOM_A` and `RANDOM_C` in `random.h` using a pre-processor macro. 
Set their values to `1103515245` and `12345` respectively. */
#define RANDOM_A 1103515245
#define RANDOM_C 12345

/* 2.5 Create a function declaration (in `random.h`) and definition (in `random.c`) for a function
`random_uint` returning an `unsigned int`. Implement a linear generator where x can use the variable `rseed`
(with an initial value x0 = RAND_SEED as per exercise 2.4), and A and C are your parameters. */
extern unsigned int random_uint();

// 3.1 Add a new function definition and declaration (random_float) returning a random float.
extern float random_float();
