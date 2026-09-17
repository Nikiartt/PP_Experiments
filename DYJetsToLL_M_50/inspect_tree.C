#include <TFile.h>
#include <TTree.h>
#include <TBranch.h>
#include <TObjArray.h>

#include <fstream>
#include <iostream>
#include <string>

void inspect_tree()
{
    std::ifstream input("dyjets_xrootd_files.txt");
    std::string file;
    std::getline(input, file); // reads only the first line

    TFile* f = TFile::Open(file.c_str());
    if (!f || f->IsZombie()) {
        std::cerr << "Could not open file." << std::endl;
        return;
    }

    TTree* t = (TTree*)f->Get("Events");
    if (!t) {
        std::cerr << "Could not find Events tree." << std::endl;
        return;
    }

    std::cout << "\nOpened remote ROOT file:\n" << file << "\n" << std::endl;

    std::cout << "Number of events in this file: " << t->GetEntries() << "\n" << std::endl;

    std::cout << "List of all columns/branches in the Events tree:\n" << std::endl;

    TObjArray* branches = t->GetListOfBranches();

    for (int i = 0; i < branches->GetEntries(); ++i) {
        TBranch* branch = (TBranch*)branches->At(i);
        TLeaf* leaf = (TLeaf*)branch->GetListOfLeaves()->At(0);

        std::cout << i << " : " << branch->GetName() << " [";

        int printed = 0;

        for (Long64_t entry = 0; entry < t->GetEntries() && printed < 3; ++entry) {
            branch->GetEntry(entry);

            for (int j = 0; j < leaf->GetLen() && printed < 3; ++j) {
                std::cout << leaf->GetValue(j);
                ++printed;

                if (printed < 3) {
                    std::cout << ", ";
                }
            }
        }

        std::cout << "]" << std::endl;
    }

    std::cout << "\nTotal number of branches: "
              << branches->GetEntries()
              << "\n" << std::endl;

    std::cout << "\nFirst 10 events with selected real data:\n" << std::endl;

    t->Scan(
        "run:luminosityBlock:event:"
        "nMuon:Muon_pt:Muon_eta:Muon_phi:Muon_charge",
        "",
        "",
        10
    );
}