#include <ROOT/RDataFrame.hxx>
#include <ROOT/RSnapshotOptions.hxx>
#include <ROOT/RDF/RSampleInfo.hxx>
#include <Compression.h>
#include <TFile.h>
#include <TTree.h>
#include <TSystem.h>

#include <algorithm>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

std::string trim(const std::string& value)
{
    const auto first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return "";

    const auto last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::string> loadSelectedFiles(
    const std::string& fileListPath,
    size_t filesPerOutput,
    size_t outputChunkIndex)
{
    std::ifstream input(fileListPath);

    if (!input)
        throw std::runtime_error("Cannot open ROOT file list: " + fileListPath);

    std::vector<std::string> allFiles;
    std::string path;

    while (std::getline(input, path)) {
        path = trim(path);

        if (!path.empty() && path[0] != '#')
            allFiles.push_back(path);
    }

    if (allFiles.empty())
        throw std::runtime_error("No ROOT files found in: " + fileListPath);

    const size_t start =
        filesPerOutput == 0
            ? 0
            : outputChunkIndex * filesPerOutput;

    if (start >= allFiles.size())
        throw std::runtime_error(
            "Chunk starts outside file list. Total files: " +
            std::to_string(allFiles.size()));

    const size_t end =
        filesPerOutput == 0
            ? allFiles.size()
            : std::min(start + filesPerOutput, allFiles.size());

    return std::vector<std::string>(
        allFiles.begin() + start,
        allFiles.begin() + end);
}

void createParentDirectory(const std::string& file)
{
    const auto pos = file.find_last_of("/\\");

    if (pos != std::string::npos)
        gSystem->mkdir(file.substr(0, pos).c_str(), true);
}

void saveSourceFileMetadata(
    const std::string& outputFile,
    const std::vector<std::string>& sourceFiles)
{
    std::unique_ptr<TFile> file(
        TFile::Open(outputFile.c_str(), "UPDATE"));

    if (!file || file->IsZombie())
        throw std::runtime_error(
            "Cannot reopen output file: " + outputFile);

    UInt_t sourceFileId = 0;
    std::string sourcePath;

    TTree sourceTree(
        "SourceFiles",
        "Original source ROOT files");

    sourceTree.Branch(
        "SourceFileId",
        &sourceFileId);

    sourceTree.Branch(
        "SourcePath",
        &sourcePath);

    for (size_t i = 0; i < sourceFiles.size(); ++i) {
        sourceFileId = static_cast<UInt_t>(i);
        sourcePath = sourceFiles[i];
        sourceTree.Fill();
    }

    sourceTree.Write("", TObject::kOverwrite);

    file->Close();
}

void saveReducedRoot(
    const std::string& fileListPath,
    const std::string& outputFile,
    size_t filesPerOutput,
    size_t outputChunkIndex,
    unsigned numberOfThreads)
{
    const auto files = loadSelectedFiles(
        fileListPath,
        filesPerOutput,
        outputChunkIndex);

    createParentDirectory(outputFile);

    std::cout
        << "\n========================================\n"
        << "       REDUCED ROOT DATA CREATION\n"
        << "========================================\n"
        << "File list:         " << fileListPath << '\n'
        << "Chunk index:       " << outputChunkIndex << '\n'
        << "Input files:       " << files.size() << '\n'
        << "Threads:           " << numberOfThreads << '\n'
        << "Output file:       " << outputFile << '\n'
        << "========================================\n";

    for (size_t i = 0; i < files.size(); ++i)
        std::cout
            << "File " << i
            << ": " << files[i] << '\n';

    if (numberOfThreads > 1)
        ROOT::EnableImplicitMT(numberOfThreads);

    ROOT::RDataFrame inputDf(
        "Events",
        files);

    auto df = inputDf.DefinePerSample(
        "SourceFileId",
        [files](
            unsigned int,
            const ROOT::RDF::RSampleInfo& info) -> UInt_t
        {
            for (UInt_t i = 0; i < files.size(); ++i) {
                if (info.Contains(files[i]))
                    return i;
            }

            throw std::runtime_error(
                "Cannot determine SourceFileId");
        });

    std::vector<std::string> columns = {
        "SourceFileId",

        "run",
        "luminosityBlock",
        "event",

        "HLT_IsoMu24",
        "MET_pt",

        "Muon_pt",
        "Muon_eta",
        "Muon_phi",
        "Muon_charge",

        "Muon_isGlobal",
        "Muon_nStations",
        "Muon_nTrackerLayers",
        "Muon_dxy",
        "Muon_ptErr",
        "Muon_tkRelIso",

        "TrigObj_id",
        "TrigObj_pt",
        "TrigObj_eta",
        "TrigObj_phi"
    };

    const std::vector<std::string> optionalColumns = {
        "nMuon",
        "nTrigObj",
        "genWeight"
    };

    std::cout
        << "\nChecking required columns...\n";

    for (const auto& column : columns) {
        if (!df.HasColumn(column))
            throw std::runtime_error(
                "Required column missing: " + column);
    }

    for (const auto& column : optionalColumns) {
        if (df.HasColumn(column))
            columns.push_back(column);
        else
            std::cout
                << "Optional column missing: "
                << column << '\n';
    }

    std::cout
        << "\nColumns to save: "
        << columns.size() << "\n\n";

    for (const auto& column : columns) {
        std::cout
            << "  "
            << column
            << " ["
            << df.GetColumnType(column)
            << "]\n";
    }

    ROOT::RDF::RSnapshotOptions options;

    options.fMode = "RECREATE";
    options.fCompressionAlgorithm = ROOT::kZSTD;
    options.fCompressionLevel = 5;

    std::cout
        << "\n========================================\n"
        << "Reading ROOT files and writing output\n"
        << "========================================\n";

    df.Snapshot(
        "Events",
        outputFile,
        columns,
        options);

    if (numberOfThreads > 1)
        ROOT::DisableImplicitMT();

    saveSourceFileMetadata(
        outputFile,
        files);

    std::unique_ptr<TFile> result(
        TFile::Open(outputFile.c_str(), "READ"));

    if (!result || result->IsZombie())
        throw std::runtime_error(
            "Cannot verify output file: " + outputFile);

    auto* tree =
        dynamic_cast<TTree*>(
            result->Get("Events"));

    if (!tree)
        throw std::runtime_error(
            "Events tree missing from output file");

    const Long64_t savedEvents =
        tree->GetEntries();

    std::cout
        << "\n========================================\n"
        << "              COMPLETED\n"
        << "========================================\n"
        << "Input files:    " << files.size() << '\n'
        << "Saved events:   " << savedEvents << '\n'
        << "Saved columns:  " << columns.size() << '\n'
        << "Output:         " << outputFile << '\n'
        << "========================================\n";

    result->Close();
}

void SaveReducedData()
{
    const unsigned NumberOfThreads = 5;

    const size_t FilesPerOutput = 1;
    const size_t OutputChunkIndex = 0;

    const std::string InputFileList =
        "dy_mc_main_xrootd_files.txt";

    const std::string OutputFile =
        "ReducedData/dy_part_0.root";

    saveReducedRoot(
        InputFileList,
        OutputFile,
        FilesPerOutput,
        OutputChunkIndex,
        NumberOfThreads);
}