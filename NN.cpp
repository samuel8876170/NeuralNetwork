#include "include/NN.h"

using namespace std;


int main(){
    vector<int> layers = {2, 2, 1};
    vector<vector<double>> x_train = {{0, 0},
                                      {0, 1},
                                      {1, 0},
                                      {1, 1}};
    vector<vector<double>> x_test = {{0, 0},
                                     {0, 1},
                                     {1, 0},
                                     {1, 1}};
    vector<vector<double>> labels = {{0},
                                     {1},
                                     {1},
                                     {0}};
    vector<string> activation_functions = {"tanh", "tanh"};

    NeuralNetwork nn;
    nn.CreateFullyDenseNetwork(layers, activation_functions);
    nn.SetTrainingSet(x_train, false);
    nn.SetLabel(labels, false);
    nn.SetLearningRate(0.1);
    nn.SetTolerance(0.001);

    do {
        nn.FeedForward();
        nn.BackPropagate();
    } while (!nn.WithinTolerance());

    nn.Predict(x_test);


    return 0;
}
