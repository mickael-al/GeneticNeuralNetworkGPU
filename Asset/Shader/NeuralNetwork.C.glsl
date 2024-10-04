#version 450

#define GROUP_SIZE 256
#define MAX_DIM_GROUPS 256
#define MAX_DIM_THREADS (GROUP_SIZE * MAX_DIM_GROUPS)
#define MAX_DIM_THREADS_THREADS (MAX_DIM_THREADS * MAX_DIM_GROUPS)

float tanh_activation(float x)
{
	return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
}

float relu(float x) 
{
	return max(0.0, x);
}

float leaky_relu(float x) 
{
	return (x > 0.0) ? x : 0.01 * x;
}

float sigmoid(float x) 
{
	return 1.0 / (1.0 + exp(-x));
}

float elu(float x) 
{
	return (x >= 0.0) ? x : (exp(x) - 1.0);
}

float swish(float x) 
{
	return x / (1.0 + exp(-x));  // x * sigmoid(x)
}

float prelu(float x, float alpha) 
{
	return (x >= 0.0) ? x : alpha * x;
}

float step_function(float x) 
{
	return (x >= 0.0) ? 1.0 : 0.0;
}

layout(std430, binding = 0) buffer NNData
{
	uint numberNN;//nombre de reseaux de neuronne
	uint sizeNN;//la taille d'un reseau de neuronne
	uint sizePPold;//la taille de perceptrons dans la couche precedente precedente
	uint sizePPrevious;//la taille de perceptrons dans la couche precedente
	uint sizePCurrent;//la taille de perceptrons dans la couche actuel	
	uint lop;//local previous offset
	uint loc;//local offset current
	uint activationType;//activation Type
}nnd;

layout(std430, binding = 1) buffer NNValue
{
	float[] data;
}nnv;

layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
	int i = int(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * MAX_DIM_THREADS + gl_GlobalInvocationID.z * MAX_DIM_THREADS_THREADS);
	
}