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

uint hash(uint x) {
	x ^= x >> 16;
	x *= 0x7feb352du;
	x ^= x >> 15;
	x *= 0x846ca68bu;
	x ^= x >> 16;
	return x;
}

float randomInRange(uint value) 
{
	uint hashedValue = hash(value);
	float normalized = float(hashedValue % 1000u) / 1000.0;
	return normalized * 2.0 - 1.0;
}

uniform uint random_uniform;

layout(local_size_x = GROUP_SIZE, local_size_y = 1, local_size_z = 1) in;
void main()
{
	uint nid = uint(gl_GlobalInvocationID.x + gl_GlobalInvocationID.y * MAX_DIM_THREADS + gl_GlobalInvocationID.z * MAX_DIM_THREADS_THREADS);
	if (nid >= nnd.sizePCurrent * nnd.numberNN)
	{
		return;
	}
	uint lpi = nid%nnd.sizePCurrent;//localPerceptronsIndex
	uint off_inn = uint(float(nid)/ float(nnd.sizePCurrent)) * nnd.numberNN;//indexNeuralNetwork l'indice qui correspond au block du reseaux de neuronne
	uint off_link = (nnd.sizePPrevious * lpi) + (2 * lpi);//l'offset de decallage des weight
	for (uint i = 0; i < nnd.sizePPrevious; i++)//weight from current layer and use size of previous layer = number weight of current layer
	{
		nnv.data[off_inn + nnd.loc + off_link + i] = randomInRange(nid+(i* nnd.sizeNN)+ random_uniform);
	}
	nnv.data[off_inn + nnd.loc + off_link + nnd.sizePPrevious] = 0.0f;// perceptrons value
	nnv.data[off_inn + nnd.loc + off_link + nnd.sizePPrevious + 1] = nnd.activationType;//fonction d'activation
	
}