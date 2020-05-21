#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdbool.h>
#include <time.h>

// Include the header file 
// Uncomment below if including into a `.c` file rather than `.cu` file
//extern "C" {
	#include "NBodyVisualiser.h"
//}

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>

#define TIMING_FRAME_COUNT 20

// User supplied globals
static unsigned int N;
static unsigned int D;
static MODE M;
const float *PositionsX = 0;
const float *PositionsY = 0;
const nbody_soa *Bodies = 0;
const float *Densities = 0;
void(*simulate_function)(void) = 0;

// Instancing variables for histogram
GLuint vao_hist = 0;
GLuint vao_hist_vertices = 0;
GLuint tbo_hist = 0;
GLuint tex_hist = 0;
GLuint vao_hist_instance_ids = 0;

// Instancing variables for nbody
GLuint vao_nbody = 0;
GLuint vao_nbody_vertices = 0;
GLuint tbo_nbody = 0;
GLuint tex_nbody = 0;
GLuint vao_nbody_instance_ids = 0;

// Mouse controls
int mouse_old_x, mouse_old_y;
int mouse_buttons = 0;
float rotate_x = 0.0, rotate_z = 0.0;
float translate_z = -1.0;

// Vertex shader handles
GLuint vs_hist_shader = 0;
GLuint vs_nbody_shader = 0;
GLuint vs_hist_program = 0;
GLuint vs_nbody_program = 0;
GLuint vs_hist_instance_index = 0;
GLuint vs_nbody_instance_index = 0;

// Render options
bool display_bodies = true;
bool display_density = false;

// Cuda graphics resources
struct cudaGraphicsResource *cuda_nbody_vbo_resource;
struct cudaGraphicsResource *cuda_hist_vbo_resource;

// Timing variables for FPS calculation
float elapsed = 0;
float prev_time = 0;
unsigned int frames;
char title[128];

// Function prototypes
void displayLoop(void);
void initHistShader();
void initNBodyShader();
void initHistVertexData();
void initNBodyVertexData();
void initGL();
void destroyViewer();
void render(void);
void checkGLError();
void handleKeyboardDefault(unsigned char key, int x, int y);
void handleMouseDefault(int button, int state, int x, int y);
void handleMouseMotionDefault(int x, int y);
void checkCUDAError(const char *msg);

// Vertex shader source code
const char* hist_vertexShaderSource =
{
    "#version 130				                            							\n"
    "#extension GL_EXT_gpu_shader4 : enable												\n"
	"uniform samplerBuffer instance_tex;												\n"
	"in uint instance_index;						        							\n"
	"void main()																		\n"
	"{																					\n"
    "	float instance_data = texelFetchBuffer(instance_tex, int(instance_index)).x;	\n"
	"	vec4 position = vec4(gl_Vertex.x, gl_Vertex.y, 0.0f, 1.0f);						\n"
	"	gl_FrontColor = vec4(instance_data, 0.0f, 0.0f, 0.0f);							\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    				\n"
	"}																					\n"
};

const char* nbody_vertexShaderSource =
{
    "#version 130										                                \n"
    "#extension GL_EXT_gpu_shader4 : enable												\n"
	"uniform samplerBuffer instance_tex;												\n"
	"in uint instance_index;									        				\n"
	"void main()																		\n"
	"{																					\n"
    "	vec2 instance_data = texelFetchBuffer(instance_tex, int(instance_index)).xy;	\n"
	"	vec4 position = vec4(gl_Vertex.x+instance_data.x,								\n"
	"					     gl_Vertex.y+instance_data.y,								\n"
	"						 gl_Vertex.z, 1.0f);										\n"
	"	gl_FrontColor = vec4(1.0f, 1.0f, 1.0f, 0.0f);									\n"
	"   gl_Position = gl_ModelViewProjectionMatrix * position;		    				\n"
	"}																					\n"
};

////////////////////////////////        CUDA Kernels       ////////////////////////////////
__global__ void copyNBodyData2f(float* buffer, const float *x, const float *y, unsigned int N) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		// Copy data to mapped `buffer`, which should have length at least `2N`
		float* ptr = &buffer[i + i]; // Locate the address of position `2i` in `buffer` for thread `i`
		ptr[0] = x[i];
		ptr[1] = y[i];
	}
}

__global__ void copyNBodyData(float* buffer, const nbody_soa* bodies, unsigned int N) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		// Copy data to mapped `buffer`, which should have length at least `2N`
		float* ptr = &buffer[i + i]; // Locate the address of position `2i` in `buffer` for thread `i`
		ptr[0] = bodies->x[i];
		ptr[1] = bodies->y[i];
	}
}

__global__ void copyHistData(float* buffer, const float* densities, unsigned int D) {
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < D*D) {
		// Copy data to mapped `buffer`, which should have length at least `D^2`
		buffer[i] = densities[i];
	}
}

//////////////////////////////// Header declared functions ////////////////////////////////
void initViewer(unsigned int n, unsigned int d, MODE m, void(*simulate)(void)) {
	N = n;
	D = d;
	M = m;
	simulate_function = simulate;

	// Check for Unified Variable Addressing (UVA) - not available in 32 bit host mode
	if (M == CUDA) {
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, 0);
		if (prop.unifiedAddressing != 1) {
			printf("Error: No Unified Variable Addressing found. Are you trying to build your CUDA code in 32bit mode?\n");
		}
	}

	// Initialise the OpenGL viewer and context
	initGL();

	// Initialise our instance rendering and the data
	initHistShader();
	initNBodyShader();
	initHistVertexData();
	initNBodyVertexData();
}

void setNBodyPositions2f(const float *positions_x, const float *positions_y) {
	if (M == CUDA){ // Check that the supplied pointers are device pointers when in `CUDA` mode
		cudaPointerAttributes attributes;

		// Host allocated memory will cause an error - check for `positions_x`
		if (cudaPointerGetAttributes(&attributes, positions_x) == cudaErrorInvalidValue){
			cudaGetLastError(); // Clear out the previous API error
			printf("Error: Pointer (positions_x) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}

		// If UVA was used, memory allocated by the device may still be `cudaMemoryTypeHost`, which can't be used by the device.
		if (attributes.type != cudaMemoryTypeDevice){
			printf("Error: Pointer (positions_x) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}

		// Host allocated memory will cause an error - check for `positions_y`
		if (cudaPointerGetAttributes(&attributes, positions_y) == cudaErrorInvalidValue){
			cudaGetLastError(); // Clear out the previous API error
			printf("Error: Pointer (positions_y) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}

		// If UVA was used, memory allocated by the device may still be `cudaMemoryTypeHost`, which can't be used by the device.
		if (attributes.type != cudaMemoryTypeDevice){
			printf("Error: Pointer (positions_y) passed to setNBodyPositions2f must be a device pointer in CUDA mode!\n");
			return;
		}
	}

	// Else we are in `CPU` or `OPENMP` mode and don't check whether host pointers were received
	PositionsX = positions_x;
	PositionsY = positions_y;
	if (Bodies != 0){
		printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
	}
}

void setNBodyPositions(const nbody_soa *bodies) {
	if (M == CUDA){ // Check that the supplied pointer is a device pointer when in `CUDA` mode
		cudaPointerAttributes attributes;

		// Host allocated memory will cause an error - check for `bodies`
		if (cudaPointerGetAttributes(&attributes, bodies) == cudaErrorInvalidValue){
			cudaGetLastError(); // Clear out the previous API error
			printf("Error: Pointer (bodies) passed to setNBodyPositions must be a device pointer in CUDA mode!\n");
			return;
		}

		// If UVA was used, memory allocated by the device may still be `cudaMemoryTypeHost`, which can't be used by the device.
		if (attributes.type != cudaMemoryTypeDevice){
			printf("Error: Pointer (bodies) passed to setNBodyPositions must be a device pointer in CUDA mode!\n");
			return;
		}
	}

	// Else we are in `CPU` or `OPENMP` mode and don't check whether a host pointer was received
	Bodies = bodies;
	if ((PositionsX != 0) || (PositionsY != 0)){
		printf("Warning: You should use either setNBodyPositions2f or setNBodyPositions\n");
	}
}

void setHistogramData(const float *densities) { // Alias function to avoid repetition of code
	setActivityMapData(densities);
}

void setActivityMapData(const float *activity) {
	if (M == CUDA){ // Check that the supplied pointer is a device pointer when in `CUDA` mode
		cudaPointerAttributes attributes;

		// Host allocated memory will cause an error - check for `activity`
		if (cudaPointerGetAttributes(&attributes, activity) == cudaErrorInvalidValue){
			cudaGetLastError(); // Clear out the previous API error
			printf("Error: Pointer passed to setActivityMap (or setHistogramData) must be a device pointer in CUDA mode!\n");
			return;
		}

		// If UVA was used, memory allocated by the device may still be `cudaMemoryTypeHost`, which can't be used by the device.
		if (attributes.type != cudaMemoryTypeDevice){
			printf("Error: Pointer passed to setActivityMap (or setHistogramData) must be a device pointer in CUDA mode!\n");
			return;
		}
	}

	// Else we are in `CPU` or `OPENMP` mode and don't check whether a host pointer was received
	Densities = activity;
}

void startVisualisationLoop() {
	glutMainLoop();
}

//////////////////////////////// Source module functions ////////////////////////////////
void displayLoop(void) {
	unsigned int i;
	float *dptr;
	size_t num_bytes;
	unsigned int blocks;
	float t;

	if (simulate_function == 0) {
		printf("Error: Simulate function has not been defined by calling initViewer(...)\n");
		return;
	}

	// Frames Per Second timing
	if (M == CUDA) {
		cudaDeviceSynchronize();
	}
	t = (float)clock(); // Take a timestamp (usually clock ticks measures milliseconds)
	if (prev_time) { // Update the elapsed time (ms) if not the first iteration of the display loop
		elapsed += t - prev_time;
	}
	prev_time = t;
	frames++; // Increment the frame counter
	if (frames == TIMING_FRAME_COUNT) { // Measure FPS and write to window title after every 20 frames (`TIMING_FRAME_COUNT`)
		frames = 0; // Reset the frames counter
		elapsed *= CLOCKS_PER_SEC; // Calculate elapsed time in seconds
		sprintf(title, "Com4521 Assignment - NBody Visualiser (%f FPS)", (float)TIMING_FRAME_COUNT / elapsed);
		glutSetWindowTitle(title);
		elapsed = 0; // Reset elapsed time
 	}

	// Call the simulation function
	simulate_function();

	// Map data from user supplied pointers into Texture Buffer Object
	if (M == CUDA) { // Map data from user supplied pointers into TBO using CUDA
		// Nbody positions data: map buffer to device pointer so a GPU kernel can populate it
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_nbody);
		num_bytes = N * 3 * sizeof(float);
		cudaGraphicsMapResources(1, &cuda_nbody_vbo_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_nbody_vbo_resource);
		// Prepare kernel launch parameters
		blocks = N / 256;
		if ((N % 256) != 0) { // 256 threads per block, ensure least number of blocks for total threads to exceed `N`
			blocks++;
		}	
		// Kernel to map data into buffer - two possible formats for users to supplier body position data
		if (Bodies != 0) {
			copyNBodyData << <blocks, 256 >> >(dptr, Bodies, N);
		}
		else if ((PositionsX != 0) && (PositionsY != 0)) {
			copyNBodyData2f << <blocks, 256 >> >(dptr, PositionsX, PositionsY, N);
		}
		cudaGraphicsUnmapResources(1, &cuda_nbody_vbo_resource, 0);
		checkCUDAError("Error copying NBody position data from supplied device pointer\n");
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

		// Histogram/activity map data: map buffer to device pointer so a GPU kernel can populate it
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_hist);
		num_bytes = D * D * sizeof(float);
		cudaGraphicsMapResources(1, &cuda_hist_vbo_resource, 0);
		cudaGraphicsResourceGetMappedPointer((void **)&dptr, &num_bytes, cuda_hist_vbo_resource);
		// Prepare kernel launch parameters
		blocks = D * D / 256;
		if (((D * D) % 256) != 0) { // 256 threads per block, ensure least number of blocks for total threads to exceed `D^2`
			blocks++;
		}
		// Kernel to map data into buffer
		copyHistData << <blocks, 256 >> >(dptr, Densities, D);
		cudaGraphicsUnmapResources(1, &cuda_hist_vbo_resource, 0);
		checkCUDAError("Error copying Activity Map data from supplied device pointer\n");
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);
	}
	else{ // Map data from user supplied pointers into Texture Buffer Object using CPU for `CPU` or `OPENMP` mode
		// Map buffer to Texture Buffer Object for Nbody positions and copy data to it from user supplied pointer
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_nbody);
		dptr = (float*)glMapBuffer(GL_TEXTURE_BUFFER_EXT, GL_WRITE_ONLY);	// `tbo_nbody` buffer
		if (dptr == 0) {
			printf("Error: Unable to map NBody Texture Buffer Object\n");
			return;
		}
		if (Bodies != 0) {
			for (i = 0; i < N; i++) {
				unsigned int index = i + i;
				dptr[index] = Bodies->x[i];
				dptr[index + 1] = Bodies->y[i];
			}
		}
		else if ((PositionsX != 0) && (PositionsY != 0)) {
			for (i = 0; i < N; i++) {
				unsigned int index = i + i;
				dptr[index] = PositionsX[i];
				dptr[index + 1] = PositionsY[i];
			}
		}
		glUnmapBuffer(GL_TEXTURE_BUFFER_EXT);
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);

		// Map histogram buffer to positions Texture Body Object and copy data to it from user supplied pointer
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, tbo_hist);
		dptr = (float*)glMapBuffer(GL_TEXTURE_BUFFER_EXT, GL_WRITE_ONLY);	// `tbo_hist` buffer
		if (dptr == 0) {
			printf("Error: Unable to map Histogram Texture Buffer Object\n");
			return;
		}
		if (Densities != 0) {
			for (i = 0; i < D * D; i++) {
				dptr[i] = Densities[i];
			}
		}
		glUnmapBuffer(GL_TEXTURE_BUFFER_EXT);
		glBindBuffer(GL_TEXTURE_BUFFER_EXT, 0);
	}

	// Render
	render();
	checkGLError();
}

void initHistShader() {
	// Histogram vertex shader
	vs_hist_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs_hist_shader, 1, &hist_vertexShaderSource, 0);
	glCompileShader(vs_hist_shader);

	// Check for errors
	GLint status;
	glGetShaderiv(vs_hist_shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Histogram Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(vs_hist_shader, 1024, &len, data);
		printf("%s", data);
	}

	// Program
	vs_hist_program = glCreateProgram();
	glAttachShader(vs_hist_program, vs_hist_shader);
	glLinkProgram(vs_hist_program);
	glGetProgramiv(vs_hist_program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: Histogram Shader Program Link Error\n");
	}

	glUseProgram(vs_hist_program);

	// Get shader variables
	vs_hist_instance_index = glGetAttribLocation(vs_hist_program, "instance_index");
	if (vs_hist_instance_index == (GLuint)-1) {
		printf("Warning: Histogram Shader program missing 'attribute in uint instance_index'\n");
	}

	glUseProgram(0);
	// Check for any errors
	checkGLError();
}

void initNBodyShader() {
	// nbody vertex shader
	vs_nbody_shader = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs_nbody_shader, 1, &nbody_vertexShaderSource, 0);
	glCompileShader(vs_nbody_shader);

	// Check for errors
	GLint status;
	glGetShaderiv(vs_nbody_shader, GL_COMPILE_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: nbody Program Shader Compilation Error\n");
		char data[1024];
		int len;
		glGetShaderInfoLog(vs_nbody_shader, 1024, &len, data);
		printf("%s", data);
	}

	// Program
	vs_nbody_program = glCreateProgram();
	glAttachShader(vs_nbody_program, vs_nbody_shader);
	glLinkProgram(vs_nbody_program);
	glGetProgramiv(vs_nbody_program, GL_LINK_STATUS, &status);
	if (status == GL_FALSE) {
		printf("ERROR: NBody Shader Program Link Error\n");
	}

	glUseProgram(vs_nbody_program);

	// Get shader variables
	vs_nbody_instance_index = glGetAttribLocation(vs_nbody_program, "instance_index");
	if (vs_nbody_instance_index == (GLuint)-1) {
		printf("Warning: nbody Program Shader program missing 'attribute in uint instance_index'\n");
	}

	glUseProgram(0);
	// Check for any errors
	checkGLError();
}

void initHistVertexData() {
	/* Vertex Array Object */
	glGenVertexArrays(1, &vao_hist); // Create our Vertex Array Object  
	glBindVertexArray(vao_hist); // Bind our Vertex Array Object so we can use it  

	/* Create a vertex buffer */
	// Create buffer object (all vertex positions normalised between -0.5 and +0.5)
	glGenBuffers(1, &vao_hist_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vao_hist_vertices);
	glBufferData(GL_ARRAY_BUFFER, D * D * 4 * 3 * sizeof(float), 0, GL_STATIC_DRAW);
	float* verts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	float quad_size = 1.0f / D;
	for (unsigned int y = 0; y < D; y++) {
		for (unsigned int x = 0; x < D; x++) {
			int offset = (D * y + x) * 3 * 4;

			float x_min = (float)x / D;
			float y_min = (float)y / D;

			// First vertex
			verts[offset + 0] = x_min - 0.5f;
			verts[offset + 1] = y_min - 0.5f;
			verts[offset + 2] = 0.0f;

			// Second vertex
			verts[offset + 3] = x_min - 0.5f;
			verts[offset + 4] = y_min + quad_size - 0.5f;
			verts[offset + 5] = 0.0f;

			// Third vertex
			verts[offset + 6] = x_min + quad_size - 0.5f;
			verts[offset + 7] = y_min + quad_size - 0.5f;
			verts[offset + 8] = 0.0f;

			// Fourth vertex
			verts[offset + 9] = x_min + quad_size - 0.5f;
			verts[offset + 10] = y_min - 0.5f;
			verts[offset + 11] = 0.0f;
		}
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer 
	glEnableVertexAttribArray(0);
	checkGLError();

	// instance index buffer
	glGenBuffers(1, &vao_hist_instance_ids);
	glBindBuffer(GL_ARRAY_BUFFER, vao_hist_instance_ids);
	glBufferData(GL_ARRAY_BUFFER, D*D * 4 * sizeof(unsigned int), 0, GL_STATIC_DRAW);
	unsigned int* ids = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (unsigned int y = 0; y < D; y++) {
		for (unsigned int x = 0; x < D; x++) {
			int index = D * y + x;
			int offset = index + index + index + index; // int offset = index * 4

			// Four vertices (a quad) have the same instance index
			ids[offset + 0] = index;
			ids[offset + 1] = index;
			ids[offset + 2] = index;
			ids[offset + 3] = index;
		}
	}

	// Map instance 
	glVertexAttribIPointer((GLuint)vs_hist_instance_index, 1, GL_UNSIGNED_INT, 0, 0); // Set up instance id attributes pointer in shader
	glEnableVertexAttribArray(vs_hist_instance_index);
	glUnmapBuffer(GL_ARRAY_BUFFER);

	//check for errors
	checkGLError();

	/* Texture buffer object */
	glGenBuffers(1, &tbo_hist);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_hist);
	glBufferData(GL_TEXTURE_BUFFER, D * D * 1 * sizeof(float), 0, GL_DYNAMIC_DRAW); // 1 `float` element in a texture buffer object for histogram density

	/* Generate texture */
	glGenTextures(1, &tex_hist);
	glBindTexture(GL_TEXTURE_BUFFER, tex_hist);
	glTexBuffer(GL_TEXTURE_BUFFER, GL_R32F, tbo_hist);

	// Create CUDA GL resource to write CUDA data to Texture Buffer Object
	if (M == CUDA) {
		cudaGraphicsGLRegisterBuffer(&cuda_hist_vbo_resource, tbo_hist, cudaGraphicsMapFlagsWriteDiscard);
	}

	// Unbind buffers
    glBindBuffer(GL_TEXTURE_BUFFER, 0);

	// Unbind VAO
	glBindVertexArray(0); // Unbind our Vertex Array Object 

	checkGLError();
}

void initNBodyVertexData() {
	/* Vertex Array Object */
	glGenVertexArrays(1, &vao_nbody);	// Create our Vertex Array Object  
	glBindVertexArray(vao_nbody);		// Bind our Vertex Array Object so we can use it  

	/* Create a vertex buffer */
	// create buffer object (all vertex positions normalised between -0.5 and +0.5)
	glGenBuffers(1, &vao_nbody_vertices);
	glBindBuffer(GL_ARRAY_BUFFER, vao_nbody_vertices);
	glBufferData(GL_ARRAY_BUFFER, N * 3 * sizeof(float), 0, GL_STATIC_DRAW);
	float* verts = (float*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (unsigned int i = 0; i < N; i++) {
		int offset = i + i + i; // int offset = i * 3

		// Vertex point
		verts[offset + 0] = -0.5f;
		verts[offset + 1] = -0.5f;
		verts[offset + 2] = 0.0f;
	}
	glUnmapBuffer(GL_ARRAY_BUFFER);
	glVertexAttribPointer((GLuint)0, 3, GL_FLOAT, GL_FALSE, 0, 0); // Set up our vertex attributes pointer 
	glEnableVertexAttribArray(0);
	checkGLError();

	// instance index buffer
	glGenBuffers(1, &vao_nbody_instance_ids);
	glBindBuffer(GL_ARRAY_BUFFER, vao_nbody_instance_ids);
	glBufferData(GL_ARRAY_BUFFER, N * 1 * sizeof(unsigned int), 0, GL_STATIC_DRAW);
	unsigned int* ids = (unsigned int*)glMapBuffer(GL_ARRAY_BUFFER, GL_WRITE_ONLY);
	for (unsigned int i = 0; i < N; i++) {
			//single vertex as it is a point
			ids[i] = i;
	}

	// Map instance 
	glVertexAttribIPointer((GLuint)vs_nbody_instance_index, 1, GL_UNSIGNED_INT, 0, 0); // Set up instance id attributes pointer in shader
	glEnableVertexAttribArray(vs_nbody_instance_index);
	glUnmapBuffer(GL_ARRAY_BUFFER);

	// Check for errors
	checkGLError();

	/* Texture Buffer Object */
	glGenBuffers(1, &tbo_nbody);
	glBindBuffer(GL_TEXTURE_BUFFER, tbo_nbody);
	glBufferData(GL_TEXTURE_BUFFER, N * 2 * sizeof(float), 0, GL_DYNAMIC_DRAW);	// 2 `float` elements in a texture buffer object for x and y position

	/* Generate texture */
	glGenTextures(1, &tex_nbody);
    glBindTexture(GL_TEXTURE_BUFFER, tex_nbody);
    glTexBuffer(GL_TEXTURE_BUFFER, GL_RG32F, tbo_nbody);

	// Create CUDA GL resource to write CUDA data to Texture Buffer Object
	if (M == CUDA) {
		cudaGraphicsGLRegisterBuffer(&cuda_nbody_vbo_resource, tbo_nbody, cudaGraphicsMapFlagsWriteDiscard);
	}

	// Unbind buffers
	glBindBuffer(GL_TEXTURE_BUFFER, 0);

	// Unbind VAO
	glBindVertexArray(0); // Unbind our Vertex Array Object 

	checkGLError();
}

void destroyViewer() {
	checkGLError();

	// Cleanup histigram VAO
	glBindVertexArray(vao_hist);
	glDeleteBuffers(1, &vao_hist_vertices);
	vao_hist_vertices = 0;
	glDeleteBuffers(1, &vao_hist_instance_ids);
	vao_hist_instance_ids = 0;
	glDeleteBuffers(1, &tbo_hist);
	tbo_hist = 0;
	glDeleteTextures(1, &tex_hist);
	tex_hist = 0;
	if (M == CUDA) {
		cudaGraphicsUnregisterResource(cuda_hist_vbo_resource);
	}
	glDeleteVertexArrays(1, &vao_hist);
	vao_hist = 0;

	// Cleanup nbody VAO
	glBindVertexArray(vao_nbody);
	glDeleteBuffers(1, &vao_nbody_vertices);
	vao_nbody_vertices = 0;
	glDeleteBuffers(1, &vao_nbody_instance_ids);
	vao_nbody_instance_ids = 0;
	glDeleteBuffers(1, &tbo_nbody);
	tbo_nbody = 0;
	glDeleteTextures(1, &tex_nbody);
	tex_nbody = 0;
	if (M == CUDA) {
		cudaGraphicsUnregisterResource(cuda_nbody_vbo_resource);
	}
	glDeleteVertexArrays(1, &vao_nbody);
	vao_nbody = 0;

	checkGLError();
}

void initGL() {
	// Specify command line argument for window name and initialise glut program with `glutInit` function
	int argc = 1;
	char * argv[] = { "Com4521 Assignment - NBody Visualiser" };
	glutInit(&argc, argv);

	// Initialise window
	glutInitDisplayMode(GLUT_RGB);
	glutInitWindowSize(WINDOW_WIDTH, WINDOW_HEIGHT);
	glutInitWindowPosition(100, 0);
	glutCreateWindow(*argv);

	// glew init (must be done after window creation for some odd reason)
	glewInit();
	if (!glewIsSupported("GL_VERSION_2_0 ")) {
		fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.\n");
		fflush(stderr);
		exit(0);
	}

	// Register default callbacks
	glutDisplayFunc(displayLoop);
	glutKeyboardFunc(handleKeyboardDefault);
	glutMotionFunc(handleMouseMotionDefault);
	glutMouseFunc(handleMouseDefault);
	glutSetOption(GLUT_ACTION_ON_WINDOW_CLOSE, GLUT_ACTION_CONTINUE_EXECUTION);

	// Default initialization
	glClearColor(0.0, 0.0, 0.0, 1.0);
	glDisable(GL_DEPTH_TEST);

	// Viewport
	glViewport(0, 0, WINDOW_WIDTH, WINDOW_HEIGHT);

	// Projection
	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
	gluPerspective(60.0, (GLfloat)WINDOW_WIDTH / (GLfloat)WINDOW_HEIGHT, 0.001, 10.0);
}

void render(void) {
	// Set view matrix and prepare for rendering
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();

	// Transformations
	glTranslatef(0.0, 0.0, translate_z);
	glRotatef(rotate_x, 1.0, 0.0, 0.0);
	glRotatef(rotate_z, 0.0, 0.0, 1.0);

	// Render the density field
	if (display_density) {
		// Attach the shader program to rendering pipeline to perform per vertex instance manipulation 
		glUseProgram(vs_hist_program);

		// Bind our Vertex Array Object (contains vertex buffers object and vertex attribute array)
		glBindVertexArray(vao_hist);

		// Bind and activate texture with instance data (held with the Texture Buffer Object)
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, tex_hist);

		// Draw the vertices with attached vertex attribute pointers
		glDrawArrays(GL_QUADS, 0, 4 * D * D);

		// Unbind the Vertex Array Object
		glBindVertexArray(0);

		// Disable the shader program and return to the fixed function pipeline
		glUseProgram(0);
	}

	// Render the n bodies
	if (display_bodies) {
		// Attach the shader program to rendering pipeline to perform per vertex instance manipulation 
		glUseProgram(vs_nbody_program);

		// Bind our Vertex Array Object (contains vertex buffers object and vertex attribute array)
		glBindVertexArray(vao_nbody);

		// Bind and activate texture with instance data (held with the Texture Buffer Object)
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_BUFFER_EXT, tex_nbody);

		// Draw the vertices with attached vertex attribute pointers
		glDrawArrays(GL_POINTS, 0, 1 * N);

		// Unbind the vertex array object
		glBindVertexArray(0);

		// Disable the shader program and return to the fixed function pipeline
		glUseProgram(0);
	}
	glutSwapBuffers();
	glutPostRedisplay();
}

void checkGLError() {
	int Error;
	if ((Error = glGetError()) != GL_NO_ERROR) {
		const char* Message = (const char*)gluErrorString(Error);
		fprintf(stderr, "OpenGL Error : %s\n", Message);
	}
}

void handleKeyboardDefault(unsigned char key, int x, int y) {
	switch (key) {
	case(27): case('q'): // Escape `Esc` key or `q` key
		// Return control to the users program to allow them to clean-up any allocated memory etc.
		glutLeaveMainLoop();
		break;
	case('b'): // `b` key to toggle display bodies
		display_bodies = !display_bodies;
		break;
	case('d'): // `d` key to toggle display activity grid map
		display_density = !display_density;
		break;
	}
}

void handleMouseDefault(int button, int state, int x, int y) {
	if (state == GLUT_DOWN) {
		mouse_buttons |= 1 << button;
	}
	else if (state == GLUT_UP) {
		mouse_buttons = 0;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void handleMouseMotionDefault(int x, int y) {
	float dx, dy;
	dx = (float)(x - mouse_old_x);
	dy = (float)(y - mouse_old_y);

	if (mouse_buttons & 1) { // Rotate with left click and mouse motion
		rotate_x += dy * 0.2f;
		rotate_z += dx * 0.2f;
	}
	else if (mouse_buttons & 4) { // Zoom out/in with right click and mouse motion up/down
		translate_z += dy * 0.01f;
	}

	mouse_old_x = x;
	mouse_old_y = y;
}

void checkCUDAError(const char* msg) {
	cudaError_t err = cudaGetLastError();
	if (err != cudaSuccess) {
		fprintf(stderr, "CUDA ERROR: %s: %s.\n", msg, cudaGetErrorString(err));
		exit(EXIT_FAILURE);
	}
}
