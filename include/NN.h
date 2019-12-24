//
// Created by Samuel on 2019/12/18.
//
#ifndef CPPSDL_NN_H
#define CPPSDL_NN_H

#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <ctime>
#include <string>

using namespace std;

default_random_engine  generator;
normal_distribution<double> normal;


//// matrix operations
vector<vector<double>> transpose(vector<vector<double>> values_set) {
    vector<vector<double>> result;
    for (unsigned int set_i = 1; set_i < values_set.size(); ++set_i) {
        if (values_set[set_i].size() != values_set[set_i - 1].size()) {
            printf("Error(transpose): Matrix is not a square\n");
            return result;
        }
    }

    for (unsigned int j = 0; j < values_set[0].size(); ++j) {
        vector<double> result1d;
        result.push_back(result1d);
        for (vector<double> &values : values_set)
            result[j].push_back(values[j]);
    }

    return result;
}
vector<double> addition(vector<double> a, vector<double> b) {
    vector<double> result;
    if (a.size() != b.size()) {
        printf("Error(addition): a.size():%d != b.size():%d\n", a.size(), b.size());
        return result;
    }
    result.reserve(a.size());
    for (unsigned int i = 0; i < a.size(); ++i)
        result.push_back(a[i] + b[i]);

    return result;
}
vector<double> subtract(vector<double> a, vector<double> b){
    vector<double> result;
    if (a.size() != b.size()){
        printf("Error(subtract): a.size():%d != b.size():%d\n", a.size(), b.size());
        return result;
    }
    result.reserve(a.size());
    for (unsigned int i = 0; i < a.size(); ++i)
        result.push_back(a[i]-b[i]);

    return result;
}
vector<double> multiply(const vector<double> a, const vector<double> b = vector<double>(), double val = 0.0) {
    vector<double> result;
    result.reserve(a.size());

    if (!b.empty() && val == 0.0) {
        if (a.size() != b.size()) {
            printf("Error(multiply): a.size():%d != b.size():%d\n", a.size(), b.size());
            return result;
        }
        for (unsigned int i = 0; i < a.size(); ++i)
            result.push_back(a[i] * b[i]);
    } else if (b.empty() && val != 0.0){
        for (double value : a)
            result.push_back(val * value);
    } else {
        printf("Error(multiply): both variable b and val is set or both not set\n");
    }

    return result;
}
double sum(const vector<double>& values){
    double sum = 0;
    for (double value : values)
        sum += value;
    return sum;
}


//// statistic operations
double mean(vector<double>& values){
    int size = 0;
    double sum = 0;
    for (double value : values) {
        sum += value;
        ++size;
    }
    return (size==0)?0:sum/size;
}
double std_dev(vector<double>& values){
    int size = 0;
    double square_error = 0, miu = mean(values);
    for (double value : values){
        ++size;
        square_error += pow(value - miu, 2);
    }
    return (size == 0)?0:sqrt(square_error / size);
}

double sigmoid(double x){
    return 1/(1+exp(-x));
}
double dsigmoid(double x){
//    printf("x = %.5f ; x*(1-x) = %.5f\n", x, x*(1-x));
    return sigmoid(x)* (1 - sigmoid(x));
}

double hyper_tan(double x) {
    return (1.0 - exp(-2 * x)) / (1.0 + exp(-2 * x));
}
double dhyper_tan(double x) {
    return (1 + x) * (1 - x);
}

double leaky_relu(double x, double alpha=0.01){
    if (x > 0)
        return x;
    else
        return alpha * x;
}
double dleaky_relu(double x, double alpha=0.01){
    if (x > 0)
        return 1;
    else
        return alpha;
}

double squared_error(double predict, double actual){
    return 0.5 * pow(predict - actual, 2);
}



struct Connection{
    double weight = normal(generator);
};
struct Neuron{
    vector<double> output{};    // ith feature from n different training set -> size() == n
    string act_fcn{};
    vector<double> activated_output{};  // output after returning from activation function
    double bias{};
    vector<double> delta{};
};


class NeuralNetwork{
private:
    vector<vector<vector<Connection>>> connections; // layer->current neuron->previous neuron
    vector<vector<Neuron>> neurons;  // layer->neuron
    unsigned int layer_size;

    vector<vector<double>> training_set;
    unsigned int training_set_size;
    vector<vector<double>> label_set;   // for supervised learning
    unsigned int label_set_size;

    double learning_rate = 0.5;
    double tolerance = 0.05;

    static void LegalActivationFunction(vector<string>& activation_functions){
        for (string & activation_function : activation_functions){
            printf("activation_function : %s\n", activation_function.c_str());
            if (activation_function != "sigmoid" || activation_function != "relu" || activation_function != "tanh")
                activation_function = "sigmoid";
        }
    }
    static double activate(double x, string& acti_fcn){
        if (acti_fcn == "sigmoid")
            return sigmoid(x);
        else if (acti_fcn == "relu")
            return leaky_relu(x);
        else if (acti_fcn == "tanh")
            return hyper_tan(x);
        else
            printf("Error(Activate): No matching activation function name!!\n");
        return -999.999;
    }
    static double deri_active(double x, string& acti_fcn){
        if (acti_fcn == "sigmoid")
            return dsigmoid(x);
        else if (acti_fcn == "relu")
            return dleaky_relu(x);
        else if (acti_fcn == "tanh")
            return dhyper_tan(x);
        else
            printf("Error(deri_active): No matching activation function name!!\n");
        return -987.6543;
    }
    static void NormalizeAll(vector<vector<double>>& data_set){
        for (auto & data : data_set) {
            double miu = mean(data), dev = std_dev(data);
            for (double & value : data) {
                if (dev != 0)
                    value = (value - miu) / dev;
            }
        }
    }

public:
    void CreateFullyDenseNetwork(vector<int>& layers, vector<string>& activat_fcns){
        activat_fcns.insert(activat_fcns.begin(), "sigmoid");
        // check legality of activation functions
        if (activat_fcns.size() != layers.size()){
            if (activat_fcns.size() < layers.size()){
                printf("Error(activate): activat_fcns.size():%d < layers.size():%d\n"
                        , activat_fcns.size(), layers.size());
                for (; layers.size() - activat_fcns.size() > 0;)
                    activat_fcns.emplace_back("sigmoid");
            } else {
                printf("Error(activate): activat_fcns.size():%d > layers.size():%d\n"
                        , activat_fcns.size(), layers.size());
                printf("Function cannot run through normally, please try again.\n");
                return;
            }
        }
        LegalActivationFunction(activat_fcns);

        // create fully connected network
        layer_size = layers.size();
        for (unsigned int layer = 0; layer < layer_size; ++layer) {
            vector<vector<Connection>> con2d;
            vector<Neuron> neu1d;
            connections.push_back(con2d);
            neurons.push_back(neu1d);

            double layer_bias = normal(generator);
            for (int neur = 0; neur < layers[layer]; ++neur) {
                vector<Connection> con1d;
                connections[layer].push_back(con1d);

                Neuron tempN{};
                tempN.bias = layer_bias;
                tempN.act_fcn = activat_fcns[layer];
                if (layer == 0) {
                    Connection tempC{};
                    tempN.bias = 0;
                    tempC.weight = 1;
                    connections[layer][neur].push_back(tempC);
                } else {
                    for (int pren = 0; pren < layers[layer - 1]; ++pren) {
                        Connection tempC{};
                        tempC.weight = normal(generator);
                        connections[layer][neur].push_back(tempC);
                    }
                }
                neurons[layer].push_back(tempN);
            }
        }
    }

    void FeedForward() {
        if (training_set.empty()) {
            printf("You Need To Input Your Training Data First! (Call : \"SetTrainingSet()\")\n");
            return;
        }

        for (unsigned int layer = 1; layer < layer_size; ++layer)
            for (unsigned int neuron = 0; neuron < neurons[layer].size(); ++neuron)
                for (unsigned int set_i = 0; set_i < training_set.size(); ++set_i) {
                    neurons[layer][neuron].output[set_i] = neurons[layer][neuron].bias;
                    for (unsigned int pren = 0; pren < connections[layer][neuron].size(); ++pren)
                        neurons[layer][neuron].output[set_i] +=
                                connections[layer][neuron][pren].weight * neurons[layer - 1][pren].activated_output[set_i];

                    neurons[layer][neuron].activated_output[set_i] =
                            activate(neurons[layer][neuron].output[set_i], neurons[layer][neuron].act_fcn);
                }
    }

    void BackPropagate(){
        if (label_set.empty()) {
            printf("You Need To Set Up Your Labels First! (Call : \"SetLabel(...)\")\n");
            return;
        }

        unsigned int last_layer = layer_size - 1;
        // back prop of last layer
        for (unsigned int neu = 0; neu < neurons[last_layer].size(); ++neu){
            vector<double> y_sub = subtract(neurons[last_layer][neu].activated_output, transpose(label_set)[neu]); //(ŷ - y)
            vector<double> dacti_a; dacti_a.reserve(training_set_size);  // dsigmoid( sum_product_output )
            for (double value : neurons[last_layer][neu].output)
                dacti_a.push_back(deri_active(value, neurons[last_layer][neu].act_fcn));

            neurons[last_layer][neu].delta = multiply(y_sub, dacti_a);  // update delta

            // update weight and bias
            neurons[last_layer][neu].bias -= learning_rate * sum(neurons[last_layer][neu].delta);
            for (unsigned pren = 0; pren < connections[last_layer][neu].size(); ++pren) {
                connections[last_layer][neu][pren].weight -= learning_rate *sum(
                        multiply(neurons[last_layer][neu].delta,neurons[last_layer - 1][pren].output));
            }
        }

        // back prop of hidden layers
        for (unsigned int layer = last_layer - 1; layer > 0; --layer){
            for (unsigned int neu = 0; neu < neurons[layer].size(); ++neu){
                vector<double> weight_delta = multiply(neurons[layer + 1][0].delta,
                                                       vector<double>(), connections[layer+1][0][neu].weight);  // sum(weight*delta)

                for (unsigned int posn = 1; posn < neurons[layer+1].size(); ++posn)
                    weight_delta = addition(weight_delta,
                            multiply(neurons[layer+1][posn].delta, vector<double>(), connections[layer+1][posn][neu].weight));

                vector<double> dsig_a; dsig_a.reserve(training_set_size);  // dsigmoid( sum_product_output )
                for (double value : neurons[layer][neu].output)
                    dsig_a.push_back(deri_active(value, neurons[layer][neu].act_fcn));

                neurons[layer][neu].delta = multiply(weight_delta, dsig_a);  // update delta

                // update weight and bias
                neurons[layer][neu].bias -= learning_rate * sum(neurons[layer][neu].delta);
                for (unsigned pren = 0; pren < connections[layer][neu].size(); ++pren)
                    connections[layer][neu][pren].weight -= learning_rate *
                            sum(multiply(neurons[layer][neu].delta, neurons[layer-1][pren].output));
            }
        }

    }

    void SetLearningRate(double lr){
        learning_rate = lr;
    }

    void SetTrainingSet(vector<vector<double>>& data_set, bool Normalize = true){
        for (unsigned int i = 0; i < data_set.size(); ++i) {
            if (data_set[i].size() != neurons[0].size()) {
                printf("Error(SetTrainingSet) : data_set[%d].size():%d != neurons[0].size():%d\n",
                        i, data_set[i].size(), neurons[0].size());
                return;
            }
        }

        this->training_set = data_set;
        training_set_size = training_set.size();
        printf(">>Size of training set: %d\n", training_set_size);

        for (vector<Neuron>& layer : neurons){
            for (Neuron& neuron : layer){
                neuron.output.clear();
                neuron.activated_output.clear();
                for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
                    double temp = 0;
                    neuron.output.push_back(temp);
                    neuron.activated_output.push_back(temp);
                    neuron.delta.push_back(temp);
                }
            }
        }

        if (Normalize){
            printf("Start normalize all training data...\n\n");
            NormalizeAll(training_set);
        }

        // initialize input neurons
        printf("Start initialize input neurons...\n\n");
        for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
            for (unsigned int data = 0; data < training_set[set_i].size(); ++data) {
                neurons[0][data].output[set_i] = training_set[set_i][data];
                neurons[0][data].activated_output[set_i] = training_set[set_i][data];
            }
        }
    }
    void SetLabel(vector<vector<double>>& labels, bool Normalize = true){
        if (training_set.empty()) {
            printf("Please set up training data first\n");
            return;
        } else if (labels.size() != training_set_size){
            printf("Error(SetLabel) : labels.size():%d != training_set_size:%d\n", labels.size(), training_set_size);
            return;
        }
        for (unsigned int i = 0; i < labels.size(); ++i) {
            if (labels[i].size() != neurons[neurons.size()-1].size()) {
                printf("Error(SetLabel) : labels[%d].size():%d != neurons[%d].size():%d\n",
                        i, labels[i].size(), neurons.size()-1, neurons[0].size());
                return;
            }
        }

        this->label_set = labels;
        label_set_size = labels.size();

        if (Normalize){
            printf("Start normalize all labels...\n\n");
            NormalizeAll(label_set);
        }
    }

    void SetTolerance(double tol){
        tolerance = tol;
    }

    bool WithinTolerance() {
        double error = 0;
        int size = 0;
        for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
            for (unsigned neu = 0; neu < neurons[layer_size - 1].size(); ++neu) {
                error += squared_error(neurons[layer_size - 1][neu].activated_output[set_i], label_set[set_i][neu]);
                ++size;
            }
        }
        error /= 2 * size;
        printf("error : %.15f\n", error);
        return error < tolerance;

//        for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
//            for (unsigned int neu = 0; neu < neurons[layer_size - 1].size(); ++neu) {
//                printf("abs(neurons[%d][%d].activated_output[%d] - label_set[%d][%d]) : %.5f\n", layer_size - 1, neu, set_i, set_i, neu,
//                       abs(neurons[layer_size - 1][neu].activated_output[set_i] - label_set[set_i][neu]));
//                if (abs(neurons[layer_size - 1][neu].activated_output[set_i] - label_set[set_i][neu]) > 1.001)
//                    return false;
//            }
//        }
//        return true;
    }

    void Predict(vector<vector<double>> &features) {
        for (vector<double> &feature : features) {
            if (feature.size() != training_set[0].size()) {
                printf("Size of prediction features does not match the training features!\n");
                return;
            }
        }

        // make predict features as activated_output of neurons at input layer
        for (unsigned int neu = 0; neu < neurons[0].size(); ++neu) {
            vector<double> temp = {};
            temp.reserve(features.size());
            for (vector<double> feature : features)
                temp.push_back(feature[neu]);
            neurons[0][neu].activated_output = temp;
        }

        // feed forward and print the result
        FeedForward();
        printf("Predict result: [");
        for (unsigned int set_i = 0; set_i < features.size(); ++set_i) {
            printf("[");
            for (Neuron & neu : neurons[layer_size - 1])
                printf("%.5f, ", neu.activated_output[set_i]);
            printf("], ");
        }
        printf("]\n");

        // change activated_output of neurons at input layer back as before
        for (Neuron neuron : neurons[0])
            // for input layer acti_out and out must be the same;
            // we didn't change output to data for prediction so we can reuse the output.
            neuron.activated_output = neuron.output;
    }


    /////////////////output function/////////////////
    void print(){
        printf("Neurons :\n");
        printf("training_set_size confirm: %d\n", neurons[0][0].activated_output.size());
        for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
            printf("  training_set %d:\n", set_i);
            for (unsigned int layer = 0; layer < layer_size; ++layer) {
                printf("    layer %d:\n    [", layer);
                for (unsigned int neur = 0; neur < neurons[layer].size(); ++neur)
                    printf("%d : %.3f | %.3f, ", neur,
                            neurons[layer][neur].activated_output.size() > neur?neurons[layer][neur].activated_output[set_i]:-9.9,
                            neurons[layer][neur].bias);
                printf("]\n");
            }
        }

        printf("Connections :\n");
        for (int layer = 0; layer < layer_size; ++layer) {
            printf("    layer %d:\n    [", layer);
            for (int pos = 0; pos < connections[layer].size(); ++pos) {
                for (int pre = 0; pre < connections[layer][pos].size(); ++pre)
                    printf("%d->%d : %.3f, ", pos, pre, connections[layer][pos][pre].weight);
            }
            printf("]\n");
        }
    }
    void print_result(){
        printf("result: [");
        for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
            printf("[");
            for (Neuron & neu : neurons[layer_size - 1])
                printf("%.5f | %.5f, ", neu.output[set_i], neu.activated_output[set_i]);
            printf("], ");
        }
        printf("]\n");
    }
    void print_squared_error(){
        printf("Final Squared Error:\n");
        for (unsigned int set_i = 0; set_i < training_set_size; ++set_i) {
            printf("    data_set[%d]: [", set_i);
            for (unsigned neui = 0; neui < neurons[layer_size - 1].size(); ++neui)
                printf("%.3f, ",
                        squared_error(neurons[layer_size-1][neui].activated_output[set_i], label_set[set_i][neui]));
            printf("]\n");
        }
    }
};


#endif //CPPSDL_NN_H
