#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TLeaf.h>
#include <TObjArray.h>

#include <fstream>
#include <iostream>
#include <iomanip>
#include <string>

void save_first_event()
{
    const char* inputFileList = "files-recid-30563.txt";
    const char* outputFileName = "first_event_full.txt";

    std::ifstream input(inputFileList);
    if (!input.is_open()) {
        std::cerr << "Could not open file list: " << inputFileList << std::endl;
        return;
    }

    std::string rootFileName;
    std::getline(input, rootFileName);

    if (rootFileName.empty()) {
        std::cerr << "File list is empty." << std::endl;
        return;
    }

    TFile* file = TFile::Open(rootFileName.c_str());
    if (!file || file->IsZombie()) {
        std::cerr << "Could not open ROOT file:" << std::endl;
        std::cerr << rootFileName << std::endl;
        return;
    }

    TTree* tree = (TTree*)file->Get("Events");
    if (!tree) {
        std::cerr << "Could not find Events tree." << std::endl;
        file->Close();
        return;
    }

    const Long64_t eventIndex = 0;
    const Long64_t numberOfEvents = tree->GetEntries();

    if (numberOfEvents == 0) {
        std::cerr << "The Events tree is empty." << std::endl;
        file->Close();
        return;
    }

    tree->GetEntry(eventIndex);

    std::ofstream output(outputFileName);
    if (!output.is_open()) {
        std::cerr << "Could not create output file: " << outputFileName << std::endl;
        file->Close();
        return;
    }

    output << std::setprecision(17);

    output << "Opened ROOT file:\n";
    output << rootFileName << "\n\n";

    output << "Tree name: Events\n";
    output << "Total number of events in file: " << numberOfEvents << "\n";
    output << "Saved event index: " << eventIndex << "\n\n";

    TObjArray* branches = tree->GetListOfBranches();
    const int numberOfBranches = branches->GetEntries();

    output << "Total number of branches: " << numberOfBranches << "\n\n";
    output << "Full data of the first event\n";
    output << "========================================\n\n";

    for (int i = 0; i < numberOfBranches; ++i) {
        TBranch* branch = (TBranch*)branches->At(i);
        if (!branch) continue;

        branch->GetEntry(eventIndex);

        TObjArray* leaves = branch->GetListOfLeaves();

        output << "Branch " << i << ": " << branch->GetName() << "\n";

        for (int l = 0; l < leaves->GetEntries(); ++l) {
            TLeaf* leaf = (TLeaf*)leaves->At(l);
            if (!leaf) continue;

            const int numberOfValues = leaf->GetNdata();

            output << "  Leaf name: " << leaf->GetName() << "\n";
            output << "  Type: " << leaf->GetTypeName() << "\n";
            output << "  Number of values in this event: " << numberOfValues << "\n";
            output << "  Values: ";

            if (numberOfValues == 0) {
                output << "[]";
            } else {
                output << "[";

                for (int j = 0; j < numberOfValues; ++j) {
                    output << leaf->GetValue(j);

                    if (j + 1 < numberOfValues) {
                        output << ", ";
                    }
                }

                output << "]";
            }

            output << "\n";
        }

        output << "\n";
    }

    output.close();
    file->Close();

    std::cout << "Saved full first event information to: "
              << outputFileName << std::endl;
}