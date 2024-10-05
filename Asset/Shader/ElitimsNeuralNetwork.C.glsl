#version 450

#define GROUP_SIZE 256
#define MAX_DIM_GROUPS 256
#define MAX_DIM_THREADS (GROUP_SIZE * MAX_DIM_GROUPS)
#define MAX_DIM_THREADS_THREADS (MAX_DIM_THREADS * MAX_DIM_GROUPS)

layout(std430, binding = 0) buffer NNData
{
	uint numberNN;//nombre de reseaux de neuronne
	uint sizeNN;//la taille d'un reseau de neuronne(le nombre de float entre chaque)
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

layout(std430, binding = 2) buffer NNIndex
{
	int[] index;
}nni;

uniform uint numberOfBest;

layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
	uint nid = uint(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * MAX_DIM_THREADS + gl_GlobalInvocationID.z * MAX_DIM_THREADS_THREADS);
	if (nid >= numberOfBest * nnd.sizeNN)
	{
		return;
	}		
	uint nnid =	uint(double(nid) / double(nnd.sizeNN));
	uint loid = nid % nnd.sizeNN;
	float data = nnv.data[nni.index[nnid]* nnd.sizeNN + loid];
	for (uint i = nnid + numberOfBest; i < nnd.numberNN; i += numberOfBest)
	{
		nnv.data[nni.index[i] * nnd.sizeNN + loid] = data;
	}
}