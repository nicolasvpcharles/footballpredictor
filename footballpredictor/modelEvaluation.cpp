#include <iostream>
#include <vector>

#include "neuralNetwork.h"
#include "datasetopen.h"

int main()
{
    // ici je recopie des parties de mon ancien code pour evaluer le modele
    //  il faut entrainer le modele
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
    // Creation du reseau neuronal
    neuralNetwork nn(data, awnser, 32, 1);

    // Entrainement du modele
    // nn.train(10000);
    // load du modele
    nn.load("neuralNetwork");

    cout << "evaluation de la precision du modele";
    // dataset de matchs je ecris des matchs de fot au hazard qui se passent en live
    vector<vector<double>> newDataset{{14, 3, 15, 8, 2, 13}, {9, 5, 7, 4, 6, 5}, {4, 6, 4, 7, 10, 4}};

    vector<double> newDatasetResponses{1, 1, 1};
    double modelEvaluation = nn.evaluate(newDataset, newDatasetResponses);
    cout << endl
         << "model evaluation " << modelEvaluation;
}