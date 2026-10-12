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
    // nn.train(10000);
    // load du modele
    nn.load("neuralNetwork");
    // test du modele
    string homeTeamName;
    // home goal scored
    double hGs;
    // home goal conceded
    double hGc;
    // home points
    double hP;
    //
    // For the away team
    //
    string awayTeamName;
    // away goal sclored
    double aGs;
    // away goal conceded
    double aGc;
    // away points
    double aP;

    // afichage terminal car j ai la fleme de faire autre chose
    cout << "\033[31m     ____  _____  _____  ____    ____    __    __    __      ____  ____  ____  ____  ____  ___  ____  _____  ____ \n"
            "    ( ___)(  _  )(  _  )(_  _)  (  _ \\  /__\\  (  )  (  )    (  _ \\(  _ \\( ___)(  _ \\(_  _)/ __)(_  _)(  _  )(  _ \\\n"
            "     )__)  )(_)(  )(_)(   )(     ) _ < /(__)\\  )(__  )(__    )___/ )   / )__)  )(_) )_)(_( (__   )(   )(_)(  )   /\n"
            "    (__)  (_____)(_____) (__)   (____/(__)(__)(____)(____)  (__)  (_)\\_)(____)(____/(____)\\___) (__) (_____)(_)\n";
    cout << endl
         << endl;
    cout << "by nicolasvpcharles on github" << endl;
    // Lit un nom d'equipe, meme s'il contient des espaces
    cout << "\033[0mEcrivez le nom de la premiere equipe\n/";
    getline(cin, homeTeamName);

    cout << "Ecrivez la moyenne de goals marques par "
         << homeTeamName << " sur les 5 derniers matchs\n/";
    cin >> hGs;

    cout << "Ecrivez la moyenne de goals concedés par "
         << homeTeamName << " sur les 5 derniers matchs\n/";
    cin >> hGc;

    cout << "Ecrivez les points obtenus par "
         << homeTeamName << " sur les 5 derniers matchs\n/";
    cin >> hP;

    // Nettoie le retour a la ligne apres les nombres
    cin.ignore(numeric_limits<streamsize>::max(), '\n');

    cout << "Ecrivez le nom de la deuxieme equipe\n/";
    getline(cin, awayTeamName);

    cout << "Ecrivez la moyenne de goals marques par "
         << awayTeamName << " sur les 5 derniers matchs\n/";
    cin >> aGs;

    cout << "Ecrivez la moyenne de goals concedés par "
         << awayTeamName << " sur les 5 derniers matchs\n/";
    cin >> aGc;

    cout << "Ecrivez les points obtenus par "
         << awayTeamName << " sur les 5 derniers matchs\n/";
    cin >> aP;
    //
    //
    //
    hGs = nn.norm(hGs / 5, 5);
    hGc = nn.norm(hGc / 5, 5);
    hP = nn.norm(hP, 5);

    // maintenant pour away
    aGs = nn.norm(aGs / 5, 5);
    aGc = nn.norm(aGc / 5, 5);
    aP = nn.norm(aP, 5);

    vector<double> predictionTestMatch = {
        hGs, hGc, hP, aGs, aGc, aP};

    double prediction = nn.predict(predictionTestMatch);

    if (nn.round(nn.round(prediction)) == 1)
    {
        cout << "le modele predit la victoire de " << homeTeamName << " a " << prediction * 100 << " %";
    }
    else
    {

        cout << "le modele predit la defete de " << homeTeamName << " a " << 100 - (prediction * 100) << " %";
    };
    // save the model
    // nn.save("neuralNetwork");
    cin >> aP;
    return 0;
}
