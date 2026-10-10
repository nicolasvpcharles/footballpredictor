#pragma once

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

inline std::vector<std::vector<double>> loadDataset(
    const std::string &filePath)
{
    std::vector<std::vector<double>> dataset;

    std::ifstream file(filePath);

    if (!file.is_open())
    {
        return dataset;
    }

    std::string line;

    // Ignore la premiere ligne contenant les noms des colonnes
    std::getline(file, line);

    while (std::getline(file, line))
    {
        if (line.empty())
        {
            continue;
        }

        std::stringstream ss(line);
        std::string value;

        std::vector<double> row;

        while (std::getline(ss, value, ','))
        {
            try
            {
                row.push_back(std::stod(value));
            }
            catch (...)
            {
                row.clear();
                break;
            }
        }

        if (!row.empty())
        {
            dataset.push_back(row);
        }
    }

    file.close();

    return dataset;
}
