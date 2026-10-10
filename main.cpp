#include <iostream>
#include <vector>

#include "neuralNetwork.h"
#include "datasetopen.h"

int main()
{
    // il faut entrainer le modele
    std::vector<std::vector<double>> awnserDataset;
    std::vector<std::vector<double>> data;

    // Chargement des deux fichiers CSV
    awnserDataset = loadDataset("dataset/result.csv");
    data = loadDataset("dataset/dataset.csv");

    // Verification du chargement
    if (data.empty() || awnserDataset.empty())
    {
        std::cout << "Erreur : un des fichiers CSV est vide ou introuvable."
                  << std::endl;
        return 1;
    }

    // Verification du nombre de matchs
    if (data.size() != awnserDataset.size())
    {
        std::cout << "Erreur : le nombre de lignes ne correspond pas."
                  << std::endl;
        std::cout << "Dataset : " << data.size() << std::endl;
        std::cout << "Resultats : " << awnserDataset.size() << std::endl;
        return 1;
    }

    // Conversion des resultats en un seul vector<double>
    std::vector<double> awnser;

    for (const auto &row : awnserDataset)
    {
        if (row.empty())
        {
            std::cout << "Erreur : une ligne de result.csv est vide."
                      << std::endl;
            return 1;
        }

        awnser.push_back(row[0]);
    }

    // Verification des 6 entrees par match
    for (const auto &row : data)
    {
        if (row.size() != 6)
        {
            std::cout << "Erreur : une ligne de dataset.csv "
                      << "ne contient pas 6 valeurs."
                      << std::endl;
            return 1;
        }
    }

    std::cout << "Nombre de matchs : " << data.size() << std::endl;
    std::cout << "Nombre de resultats : " << awnser.size() << std::endl;

    // Creation du reseau neuronal
    neuralNetwork nn(data, awnser, 16, 1);

    // Entrainement du modele
    // nn.train(10000);
    // load du modele
    nn.load("neuralNetwork");
    // test du modele
    vector<double> predictionTestMatch = {
        0.44, 0.12, 1, 0.28, 0.20, 0.53};

    double prediction = nn.predict(predictionTestMatch);

    cout << nn.round(prediction) << endl;
    cout << prediction * 100;
    // save the model
    // nn.save("neuralNetwork");
    return 0;
}