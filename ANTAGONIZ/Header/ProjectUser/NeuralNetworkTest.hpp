#ifndef __NEURAL_NETWORK_TEST__
#define __NEURAL_NETWORK_TEST__

#include "Behaviour.hpp"
#include "Debug.hpp"
#include "Engine.hpp"
#include "PointeurClass.hpp"

using namespace Ge;
namespace Ge
{
    class NeuralNetwork;
}
class NeuralNetworkTest : public Behaviour
{
public:
    void start();
    void fixedUpdate();
    void update();
    void stop();
    void onGUI();
    inline std::string serialize()
    {
        return JS::serializeStruct(*this);
    }
    inline void load(std::string jsonFile)
    {
        JS::ParseContext context(jsonFile);
        context.parseTo(*this);
    }
private:
    const ptrClass* m_pc;
    NeuralNetwork* m_neuralN;
public:
    int numberNeuralNetwork = 100;
};

REGISTER(Behaviour, NeuralNetworkTest, numberNeuralNetwork);

#endif //!__NEURAL_NETWORK_TEST__