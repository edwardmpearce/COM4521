// Header guards prevent the contents of the header from being defined multiple times where there are circular dependencies
#ifndef __NBODY_VIEWER_HEADER__
#define __NBODY_VIEWER_HEADER__

// OpenGL Graphics includes
#define WINDOWS_LEAN_AND_MEAN
#define NOMINMAX

#include <windows.h>
#include <GL/glew.h>
#include <GL/freeglut.h>

#include "NBody.h"

#define WINDOW_WIDTH 1024		// Window width
#define WINDOW_HEIGHT 768		// Window height
#define REFRESH_DELAY 10		// Refresh delay controls how frequently the scene should be re-drawn (measured in ms)

/**
 * This `NBodyVisualiser` module can be used for visualising N-body systems and activity maps. 
 * A user is required to call the following functions:
 *	1) initViewer(...) - This will initialise any data and memory required by the visualiser
 *  2) Either setNBodyPositions2f(...) or setNBodyPositions(...) depending on how your `nbody` data is stored. 
 *     These pointers are used to update the rendered positions of the bodies
 *  3) setActivityMapData(...) - This pointer will be used to update the render colours of the activity map.
 *  4) startVisualisationLoop() - This will enter a loop which will call your `simulate` function and redraw the display
 */

/** initViewer
	* initViewer must be the first call to this module. It will configure and allocate any data required for the visualiser.
	* @param	N	The NBody population size
	* @param	D	The width/height of the activity map grid
	* @param	M	The simulation mode. This must be `CPU` or `OpenMP` for Part 1 of the assignment
	* @param	simulate	A function pointer to your simulation step function which must have a `void` argument and `void` return.
	*/
void initViewer(unsigned int N, unsigned int D, MODE m, void (*simulate)(void));

/** setNBodyPositions2f
	* A user should pass a pointer to the NBody position data using either this function or `setNBodyPositions`
	* @param	positions_x	A pointer to a float array of length `N` containing the `x` position of bodies
	* @param	positions_y	A pointer to a float array of length `N` containing the `y` position of bodies
	*/
void setNBodyPositions2f(const float *positions_x, const float *positions_y);

/** setNBodyPositions
	* A user should pass a pointer to the NBody position data using either this function or `setNBodyPositions2f`
	* @param	positions	A pointer to an array of `nbody` structures containing all `N` bodies
	*/
void setNBodyPositions(const nbody *positions);

/** setActivityMapData or setHistogramData
	* A user should pass a pointer to the activity map data using either of these functions. 
	* These both perform the same operation but are both included due to the 
	* interchangeable use of the term activity map and histogram within the assignment document.
	* @param	densities	A pointer to a `float` array containing `D*D` activity values
	*/
void setActivityMapData(const float *activity);
void setHistogramData(const float *densities);

/** startVisualisationLoop
	* Starts the main visualisation loop which will send calls to your `simulate` function
	*/
void startVisualisationLoop();

#endif // __NBODY_VIEWER_HEADER__
