#define RAND_SEED 123
/* Parameters for a simple linear congruential generator defined using a pre-processor macro.
See https://en.wikipedia.org/wiki/Linear_congruential_generator for details. */
#define RANDOM_A 1103515245
#define RANDOM_C 12345

// External function declarations
extern void init_random();
extern unsigned short random_ushort();
extern unsigned int random_uint();
extern float random_float();
