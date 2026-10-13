#pragma once

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cmath>
#include <random>
#include <string>

using namespace std;

// for calculating a neural network it is n1 + w1 + b
class neuralNetwork
{
public:
    vector<vector<double>> inputLayer;
    vector<double> awnsers;
    vector<vector<double>> weight;
    vector<double> bias;

    // poids entre la couche cachee et la sortie
    vector<double> outputWeight;

    // bias de la sortie
    double outputBias;

    double outputLayer;
    double nNeurons;
    double neuronDepth;

    // fonction relu
    neuralNetwork(vector<vector<double>> nInputlayer, vector<double> nAwnsers, double numberNeurons, double nNeuronDepth)
    {
        inputLayer = nInputlayer;
        awnsers = nAwnsers;
        nNeurons = numberNeurons;
        neuronDepth = nNeuronDepth;
    };

    double randomWeight()
    {
        static random_device rd;
        static mt19937 gen(rd());
        static uniform_real_distribution<double> dist(-1.0, 1.0);

        return dist(gen);
    }

    double ReLU(double n)
    {
        if (n < 0.0)
        {
            return 0.0;
        }
        else
        {
            return n;
        };
    };

    //====================================
    // calculate loss
    //====================================
    double loss(double prediction, double awnser)
    {

        return ((prediction - awnser) * (prediction - awnser));
    };
    //==================================
    // Gradient descent
    //====================================
    double gradientDescent(double weight, double learningRate, double gradient)
    {
        weight = weight - learningRate * gradient;
        return weight;
    };

    //=====================================
    // Multiplication de matrices
    //=====================================
    vector<vector<double>> multiplyMatrices(
        vector<vector<double>> A,
        vector<vector<double>> B)
    {
        vector<vector<double>> result(
            A.size(),
            vector<double>(B[0].size(), 0));

        for (int i = 0; i < A.size(); i++)
        {
            for (int j = 0; j < B[0].size(); j++)
            {
                for (int k = 0; k < B.size(); k++)
                {
                    result[i][j] += A[i][k] * B[k][j];
                }
            }
        }

        return result;
    };
    //
    // FONCTION SIGMOIDE
    //
    double sigmoid(double x)
    {
        return 1.0 / (1.0 + exp(-x));
    };

    //
    // FONCTION TANH
    //
    double tanhActivation(double x)
    {
        return tanh(x);
    };

    //
    // DERIVE TANH
    //
    double tanhDerivative(double x)
    {
        return 1.0 - x * x;
    };

    //
    // DERIVE SIGMOIDE
    //
    double sigmoidDerivative(double x)
    {
        return x * (1.0 - x);
    };

    double norm(double value, double maxValue)
    {
        return value / maxValue;
    }

    bool round(double neuralNetworkPrediction)
    {
        if (neuralNetworkPrediction >= 0.5)
        {
            return true;
        }
        else
        {
            return false;
        };
    };

    //=======================================
    // Fontion d entrainement
    //=======================================
    void train(int eproach = 1000, double learningRate = 0.01)
    {
        // on entraine ici le reseaux de neurone
        // il faut donner un weight aleatoire pour commencer l entrainement

        bias.clear();
        weight.clear();
        outputWeight.clear();

        // on cree les neurones
        for (int i = 0; i < nNeurons; i++)
        {
            vector<double> neuronWeights;

            // chaque neurone a un weight pour chaque input
            for (int j = 0; j < inputLayer[0].size(); j++)
            {
                neuronWeights.push_back(randomWeight());
            }

            weight.push_back(neuronWeights);
            bias.push_back(randomWeight());

            // poids entre le neurone cache et la sortie
            outputWeight.push_back(randomWeight());
        }

        // bias de la sortie
        outputBias = randomWeight();

        // normalement on a des weight alleatoires
        // maintenant il va faloir lire une documentation pour comprendre comment le tout marche
        // resources
        // https://www.freecodecamp.org/news/neural-networks-explained-simply-in-python/

        int i = 0;

        while (i < eproach)
        {
            int a = 0;

            while (a < inputLayer.size())
            {
                // on parcourt chaque neurone
                int neuron = 0;

                //========================================
                // PREMIERE COUCHE
                //========================================

                vector<double> hiddenOutput;

                while (neuron < nNeurons)
                {
                    // on commence avec le bias du neurone
                    double prediction = bias[neuron];

                    // on calcule la prediction
                    int n = 0;

                    while (n < inputLayer[a].size())
                    {
                        prediction = prediction +
                                     (inputLayer[a][n] * weight[neuron][n]);

                        n = n + 1;
                    }

                    // fonction d activation tanh
                    prediction = tanhActivation(prediction);

                    hiddenOutput.push_back(prediction);

                    neuron = neuron + 1;
                }

                //========================================
                // DEUXIEME COUCHE
                //========================================

                double output = outputBias;

                neuron = 0;

                while (neuron < nNeurons)
                {
                    output = output +
                             (hiddenOutput[neuron] * outputWeight[neuron]);

                    neuron = neuron + 1;
                }

                // fonction sigmoid pour avoir une prediction entre 0 et 1
                double prediction = sigmoid(output);

                // on calcule l'erreur
                double error = prediction - awnsers[a];

                //========================================
                // BACKPROPAGATION
                //========================================

                // gradient de la sortie
                double outputGradient =
                    error * sigmoidDerivative(prediction);

                // correction des poids de sortie
                neuron = 0;
                vector<double> oldOutputWeight = outputWeight;
                while (neuron < nNeurons)
                {
                    double gradient =
                        outputGradient * hiddenOutput[neuron];

                    outputWeight[neuron] =
                        outputWeight[neuron] -
                        learningRate * gradient;

                    neuron = neuron + 1;
                }

                // correction du bias de sortie
                outputBias =
                    outputBias -
                    learningRate * outputGradient;

                //========================================
                // CORRECTION COUCHE CACHEE
                //========================================

                neuron = 0;

                while (neuron < nNeurons)
                {
                    double hiddenGradient =
                        outputGradient *
                        oldOutputWeight[neuron] *
                        tanhDerivative(hiddenOutput[neuron]);

                    // on corrige les weights
                    int n = 0;

                    while (n < inputLayer[a].size())
                    {
                        weight[neuron][n] =
                            weight[neuron][n] -
                            learningRate *
                                hiddenGradient *
                                inputLayer[a][n];

                        n = n + 1;
                    }

                    // on corrige le bias
                    bias[neuron] =
                        bias[neuron] -
                        learningRate * hiddenGradient;

                    neuron = neuron + 1;
                }

                a = a + 1;
            }

            i = i + 1;
            cout << i << "/" << eproach << endl;
        }
    };
    //==========================================================
    // FONCTION SAVE
    //==========================================================
    void save(string filePath)
    {
        // sauvegarder les weights
        ofstream weightFile(filePath + "/weight.txt");

        if (!weightFile.is_open())
        {
            cout << "error impossible to open weight.txt" << endl;
            return;
        }

        for (int i = 0; i < weight.size(); i++)
        {
            for (int j = 0; j < weight[i].size(); j++)
            {
                if (j > 0)
                {
                    weightFile << ",";
                }

                weightFile << weight[i][j];
            }

            weightFile << ";";
        }

        weightFile.close();

        // sauvegarder les bias
        ofstream biasFile(filePath + "/bias.txt");

        if (!biasFile.is_open())
        {
            cout << "error impossible to open bias.txt" << endl;
            return;
        }

        for (int i = 0; i < bias.size(); i++)
        {
            if (i > 0)
            {
                biasFile << ",";
            }

            biasFile << bias[i];
        }

        biasFile.close();

        // sauvegarder les poids de sortie
        ofstream outputWeightFile(filePath + "/outputWeight.txt");

        if (!outputWeightFile.is_open())
        {
            cout << "error impossible to open outputWeight.txt" << endl;
            return;
        }

        for (int i = 0; i < outputWeight.size(); i++)
        {
            if (i > 0)
            {
                outputWeightFile << ",";
            }

            outputWeightFile << outputWeight[i];
        }

        outputWeightFile.close();

        // sauvegarder le bias de sortie
        ofstream outputBiasFile(filePath + "/outputBias.txt");

        if (!outputBiasFile.is_open())
        {
            cout << "error impossible to open outputBias.txt" << endl;
            return;
        }

        outputBiasFile << outputBias;
        outputBiasFile.close();

        cout << "Modele sauvegarde !" << endl;
    }

    //
    // FONCTION LOAD
    //
    void loadWeight(string filePath)
    {
        ifstream weightFile(filePath + "/weight.txt");

        if (!weightFile.is_open())
        {
            cout << "error impossible to open weight.txt" << endl;
            return;
        }

        weight.clear();

        string neuronSTR;

        while (getline(weightFile, neuronSTR, ';'))
        {
            if (neuronSTR.empty())
            {
                continue;
            }

            vector<double> neuronWeights;
            stringstream weightStream(neuronSTR);
            string weightSTR;

            while (getline(weightStream, weightSTR, ','))
            {
                if (!weightSTR.empty())
                {
                    neuronWeights.push_back(stod(weightSTR));
                }
            }

            weight.push_back(neuronWeights);
        }

        weightFile.close();
    }

    //
    // Load Bias
    //

    void loadBias(string filePath)
    {
        ifstream biasFile(filePath + "/bias.txt");

        if (!biasFile.is_open())
        {
            cout << "error impossible to open bias.txt" << endl;
            return;
        }

        bias.clear();

        string biasSTR;

        while (getline(biasFile, biasSTR, ','))
        {
            if (!biasSTR.empty())
            {
                bias.push_back(stod(biasSTR));
            }
        }

        biasFile.close();
    }

    //
    // Load Output Weight
    //
    void loadOutputWeight(string filePath)
    {
        ifstream outputWeightFile(filePath + "/outputWeight.txt");

        if (!outputWeightFile.is_open())
        {
            cout << "error impossible to open outputWeight.txt" << endl;
            return;
        }

        outputWeight.clear();

        string weightSTR;

        while (getline(outputWeightFile, weightSTR, ','))
        {
            if (!weightSTR.empty())
            {
                outputWeight.push_back(stod(weightSTR));
            }
        }

        outputWeightFile.close();
    }

    //
    // Load Output Bias
    //
    void loadOutputBias(string filePath)
    {
        ifstream outputBiasFile(filePath + "/outputBias.txt");

        if (!outputBiasFile.is_open())
        {
            cout << "error impossible to open outputBias.txt" << endl;
            return;
        }

        outputBiasFile >> outputBias;

        outputBiasFile.close();
    }

    void load(string filepath)
    {
        loadWeight(filepath);
        loadBias(filepath);
        loadOutputWeight(filepath);
        loadOutputBias(filepath);
        return;
    };

    double predict(vector<double> input)
    {
        // on parcourt les neurones de la couche cachee
        vector<double> hiddenOutput;

        int neuron = 0;

        while (neuron < nNeurons)
        {
            double prediction = bias[neuron];

            int n = 0;

            while (n < input.size())
            {
                prediction = prediction +
                             (input[n] * weight[neuron][n]);

                n = n + 1;
            }

            // activation tanh
            prediction = tanhActivation(prediction);

            hiddenOutput.push_back(prediction);

            neuron = neuron + 1;
        }

        //========================================
        // SORTIE
        //========================================

        double output = outputBias;

        neuron = 0;

        while (neuron < nNeurons)
        {
            output = output +
                     (hiddenOutput[neuron] *
                      outputWeight[neuron]);

            neuron = neuron + 1;
        }

        // sigmoid
        double prediction = sigmoid(output);

        return prediction;
    };

    double evaluate(
        vector<vector<double>> trainingDataset,
        vector<double> trainingResponseDataset)
    {
        if (trainingDataset.empty() ||
            trainingDataset.size() != trainingResponseDataset.size())
        {
            cout << "Erreur : dataset invalide." << endl;
            return 0.0;
        }

        size_t i = 0;
        int accuracy = 0;

        while (i < trainingDataset.size())
        {
            double testPrediction = predict(trainingDataset[i]);
            testPrediction = round(testPrediction);

            if (testPrediction == trainingResponseDataset[i])
            {
                accuracy++;
            }

            i++;
        }

        return (static_cast<double>(accuracy) /
                trainingResponseDataset.size()) *
               100.0;
    }
};
