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
    // Creation du reseau neuronal
    neuralNetwork nn(data, awnser, 32, 1);

    // Entrainement du modele
    nn.train(5000);
    nn.save("neuralNetwork");
    cout << "model trained press any key to quitt";
    string a;
    cin >> a;
}