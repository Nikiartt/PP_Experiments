#ifndef ML_DY_C
#define ML_DY_C

#include <ROOT/RDataFrame.hxx>
#include <TMVA/Config.h>
#include <TMVA/DataLoader.h>
#include <TMVA/Factory.h>
#include <TMVA/Reader.h>
#include <TMVA/Tools.h>
#include <TMVA/Types.h>
#include <TMVA/DecisionTree.h>
#include <TMVA/Event.h>
#include <TMVA/MethodBDT.h>
#include <TCut.h>
#include <TFile.h>
#include <TRandom.h>
#include <TTree.h>
#include <TCanvas.h>
#include <TGraph.h>
#include <TLegend.h>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <ROOT/RVec.hxx>
#include <Math/Vector4D.h>
#include <TH1D.h>
#include <TSystem.h>
#include <unordered_set>
#include <random>

struct BdtHistory {
    std::vector<double> accuracy;
    std::vector<double> loss;
    std::vector<double> precision;
};

BdtHistory calculateBdtHistory(TMVA::MethodBDT& bdt, TMVA::Types::ETreeType dataType)
{
    const auto& forest = bdt.GetForest();
    const auto& events = bdt.GetEventCollection(dataType);
    const size_t nTrees = forest.size();
    BdtHistory result{std::vector<double>(nTrees, 0.0), std::vector<double>(nTrees, 0.0), std::vector<double>(nTrees, 0.0)};
    std::vector<double> correct(nTrees, 0.0), losses(nTrees, 0.0), total(nTrees, 0.0), truePositive(nTrees, 0.0), falsePositive(nTrees, 0.0);

    for (const TMVA::Event* event : events) {
        const bool signal = bdt.DataInfo().IsSignal(event);
        double score = 0.0;
        for (size_t tree = 0; tree < nTrees; ++tree) {
            score += forest[tree]->CheckEvent(event, false);
            const bool predictedSignal = score >= 0.0;
            const double probability = std::clamp(1.0 / (1.0 + std::exp(-2.0 * score)), 1e-15, 1.0 - 1e-15);
            total[tree] += 1.0;
            if (predictedSignal == signal) correct[tree] += 1.0;
            if (predictedSignal) { if (signal) truePositive[tree] += 1.0; else falsePositive[tree] += 1.0; }
            losses[tree] += signal ? -std::log(probability) : -std::log(1.0 - probability);
        }
    }

    for (size_t tree = 0; tree < nTrees; ++tree) {
        if (total[tree] > 0.0) { result.accuracy[tree] = correct[tree] / total[tree]; result.loss[tree] = losses[tree] / total[tree]; }
        const double selected = truePositive[tree] + falsePositive[tree];
        if (selected > 0.0) result.precision[tree] = truePositive[tree] / selected;
    }
    return result;
}

// void saveBdtHistory(TMVA::MethodBDT& bdt, const std::string& fileName)
// {
//     const auto h = calculateBdtHistory(bdt, TMVA::Types::kTesting);

//     TFile file(fileName.c_str(), "RECREATE");
//     gSystem->mkdir("Images", true);

//     auto save = [&](const char* name, const char* title, const std::vector<double>& y) {
//         TGraph graph(y.size());

//         for (size_t i = 0; i < y.size(); ++i)
//             graph.SetPoint(i, i + 1, y[i]);

//         graph.SetName(name);
//         graph.SetTitle(title);
//         graph.SetLineWidth(2);
//         graph.Write();

//         TCanvas canvas(("c_" + std::string(name)).c_str(), "", 900, 650);
//         graph.Draw("AL");
//         canvas.SaveAs(("Images/" + std::string(name) + ".png").c_str());
//     };

//     save("precision", "Precision vs number of trees;Trees;Precision", h.precision);
//     save("accuracy",  "Accuracy vs number of trees;Trees;Accuracy", h.accuracy);
//     save("loss",      "Log-loss vs number of trees;Trees;Loss", h.loss);

//     file.Write();
// }


void saveBdtHistory(TMVA::MethodBDT& bdt, const std::string& fileName)
{
    const auto h = calculateBdtHistory(bdt, TMVA::Types::kTesting);

    TFile file(fileName.c_str(), "RECREATE");
    gSystem->mkdir("Images", true);

    auto save = [&](const char* name, const char* title, const std::vector<double>& y) {
        TGraph graph(y.size());

        for (size_t i = 0; i < y.size(); ++i)
            graph.SetPoint(i, i + 1, y[i]);

        graph.SetName(name);
        graph.SetTitle(title);
        graph.SetLineWidth(2);
        graph.Write();

        TCanvas canvas(("c_" + std::string(name)).c_str(), "", 900, 650);
        graph.Draw("AL");
        canvas.SaveAs(("Images/" + std::string(name) + ".png").c_str());
    };

    save("precision", "Precision vs number of trees;Trees;Precision", h.precision);
    save("accuracy",  "Accuracy vs number of trees;Trees;Accuracy", h.accuracy);
    save("loss",      "Log-loss vs number of trees;Trees;Loss", h.loss);

    file.Write();
}

std::string trainDrellYanBdt(
    const std::vector<std::string>& features,
    ROOT::RDF::RNode trainingDf,
    ROOT::RDF::RNode testingDf,
    const std::string& tmvaOutputFile,
    const std::string& historyRootFile,
    const std::string& factoryName,
    const std::string& methodName,
    const std::string& datasetName,
    unsigned numberOfTrees,
    unsigned maximumDepth,
    double minimumNodeSizePercent,
    double baggedSampleFraction,
    unsigned numberOfCuts,
    double shrinkage,
    unsigned randomSeed,
    double datasetUsagePercentage,
    unsigned numberOfThreads)
{
    std::cout << "trainDrellYanBdt start\n";
    std::cout << "Features: " << features.size() << "\n";
    std::cout << "Dataset usage: " << datasetUsagePercentage << "%\n";
    std::cout << "Dataset: " << datasetName << ", Factory: " << factoryName << ", Method: " << methodName << "\n";

    if (datasetUsagePercentage <= 0.0 || datasetUsagePercentage > 100.0)
        throw std::runtime_error("Dataset usage percentage must be in (0, 100]");

    const ULong64_t keepThreshold = static_cast<ULong64_t>(std::llround(datasetUsagePercentage * 100.0));

    std::cout << "Dataset keep threshold: " << keepThreshold << "/10000\n";

    const auto useEvent = [keepThreshold, randomSeed](ULong64_t index) {
        return (index * 7919ULL + randomSeed) % 10000ULL < keepThreshold;
    };

    std::cout << "Applying dataset percentage filter to training and testing dataframes\n";
    auto trainingUsed = trainingDf.Filter(useEvent, {"Index"});
    auto testingUsed = testingDf.Filter(useEvent, {"Index"});

    std::cout << "Initializing TMVA\n";
    TMVA::Tools::Instance();//Initialize TMVA and define the input features

    TMVA::gConfig().GetIONames().fWeightFileDirPrefix = "MlTemp";//Then locations for the generated model weights are configured
    TMVA::gConfig().GetIONames().fWeightFileDir = "weights";

    std::cout << "Creating TMVA DataLoader: " << datasetName << "\n";
    TMVA::DataLoader loader(datasetName.c_str());//DataLoader is essentially TMVA's container. Holds features, signal training events, background training events, signal testing events, background testing events
    //The testing events are loaded into TMVA::DataLoader mainly so that you can evaluate the trained model on unseen data afterward

    std::cout << "Registering TMVA input features\n";
    for (const std::string& feature : features)
        loader.AddVariable(feature.c_str(), 'F');//Feature registration, 'F' means floating-point variable.

    std::cout << "Checking that all requested features exist\n";
    for (const auto& feature : features) {
        if (!trainingUsed.HasColumn(feature))
            throw std::runtime_error("Missing training feature: " + feature);
        if (!testingUsed.HasColumn(feature))
            throw std::runtime_error("Missing testing feature: " + feature);
    }

    logLine("");
    logLine("--- BDT TRAINING FEATURES ---");
    logValue("Number of features", features.size());

    for (size_t i = 0; i < features.size(); ++i)
        logValue("Feature " + std::to_string(i + 1), features[i]);

    if (!trainingUsed.HasColumn("IsDyProcess") ||
        !testingUsed.HasColumn("IsDyProcess"))
        throw std::runtime_error("Missing IsDyProcess column");

    std::cout << "IsDyProcess label column exists\n";
    std::cout << "Creating lazy Take actions for feature columns\n";

    std::vector<ROOT::RDF::RResultPtr<std::vector<Float_t>>> trainingColumns;
    std::vector<ROOT::RDF::RResultPtr<std::vector<Float_t>>> testingColumns;
    trainingColumns.reserve(features.size());
    testingColumns.reserve(features.size());

    for (const auto& feature : features) {
        trainingColumns.emplace_back(trainingUsed.Take<Float_t>(feature));
        testingColumns.emplace_back(testingUsed.Take<Float_t>(feature));
    }

    std::cout << "Creating lazy Take actions for labels\n";
    auto trainingLabels = trainingUsed.Take<bool>("IsDyProcess");
    auto testingLabels = testingUsed.Take<bool>("IsDyProcess");

    std::vector<ROOT::RDF::RResultHandle> handles;
    handles.reserve(features.size() * 2 + 2);

    for (auto& column : trainingColumns)
        handles.emplace_back(column);
    for (auto& column : testingColumns)
        handles.emplace_back(column);

    handles.emplace_back(trainingLabels);
    handles.emplace_back(testingLabels);

    std::cout << "Prepared " << handles.size() << " RDataFrame lazy actions\n";
    std::cout << "Executing training/testing RDataFrame graphs\n";

    logElapsedTime("lazy RDataFrame computations START");
    ROOT::RDF::RunGraphs(handles);//ROOT to execute several lazy RDataFrame computations together
    logElapsedTime("lazy RDataFrame computations END");

    std::cout << "RDataFrame graphs completed\n";
    std::cout << "Training rows after dataset percentage filter: " << trainingLabels->size() << "\n";
    std::cout << "Testing rows after dataset percentage filter: " << testingLabels->size() << "\n";

    ULong64_t trainingSignal = 0;
    ULong64_t trainingBackground = 0;
    ULong64_t testingSignal = 0;
    ULong64_t testingBackground = 0;

    std::cout << "Preparing function for transferring events into TMVA DataLoader\n";

    auto addEventsToTmva = [&](auto& columns,
                               auto& labels,
                               bool training) {
        std::cout << "Adding " << (training ? "training" : "testing")
                  << " events to TMVA, input rows: " << labels->size() << "\n";

        std::vector<Double_t> eventValues(features.size());

        for (size_t entry = 0; entry < labels->size(); ++entry) {
            bool valid = true;

            for (size_t i = 0; i < features.size(); ++i) {
                const Float_t value = columns[i]->at(entry);

                if (!std::isfinite(value)) {
                    valid = false;
                    break;
                }

                eventValues[i] = value;
            }

            if (!valid)
                continue;

            const bool isDyProcess = labels->at(entry);

            if (training && isDyProcess) {
                loader.AddSignalTrainingEvent(eventValues, 1.0);
                ++trainingSignal;
            } else if (training) {
                loader.AddBackgroundTrainingEvent(eventValues, 1.0);
                ++trainingBackground;
            } else if (isDyProcess) {
                loader.AddSignalTestEvent(eventValues, 1.0);
                ++testingSignal;
            } else {
                loader.AddBackgroundTestEvent(eventValues, 1.0);
                ++testingBackground;
            }
        }

        std::cout << "Finished adding " << (training ? "training" : "testing") << " events to TMVA\n";
    };

    logElapsedTime("addEventsToTmva START");

    std::cout << "Transferring training events to TMVA\n";
    addEventsToTmva(trainingColumns, trainingLabels, true);

    std::cout << "Transferring testing events to TMVA\n";
    addEventsToTmva(testingColumns, testingLabels, false);

    logElapsedTime("addEventsToTmva END");

    const ULong64_t totalTrainingEvents = trainingSignal + trainingBackground;

    std::cout << "TMVA event loading completed\n";
    std::cout << "Training signal: " << trainingSignal << "\n";
    std::cout << "Training background: " << trainingBackground << "\n";
    std::cout << "Testing signal: " << testingSignal << "\n";
    std::cout << "Testing background: " << testingBackground << "\n";

    logValue("Training events total", totalTrainingEvents);
    logLine("");
    logLine("--- BDT TRAINING DATA ---");
    logValue("Dataset usage (%)", datasetUsagePercentage);
    logValue("Training signal", trainingSignal);
    logValue("Training background", trainingBackground);
    logValue("Testing signal", testingSignal);
    logValue("Testing background", testingBackground);

    //Calculate the baseline accuracy
    const double trainingBaselineAccuracy =
        static_cast<double>(trainingBackground) /
        static_cast<double>(trainingSignal + trainingBackground);

    const double testingBaselineAccuracy =
        static_cast<double>(testingBackground) /
        static_cast<double>(testingSignal + testingBackground);

    std::cout << "Training baseline accuracy: " << trainingBaselineAccuracy << "\n";
    std::cout << "Testing baseline accuracy: " << testingBaselineAccuracy << "\n";

    logLine("");
    logLine("--- BASELINE ---");
    logValue("Training baseline accuracy (always background)", trainingBaselineAccuracy);
    logValue("Testing baseline accuracy (always background)", testingBaselineAccuracy);

    if (trainingSignal == 0 || trainingBackground == 0)
        throw std::runtime_error("Training sample does not contain both classes");//small validation

    if (testingSignal == 0 || testingBackground == 0)
        throw std::runtime_error("Testing sample does not contain both classes");

    std::cout << "Training/testing class validation passed\n";
    std::cout << "Preparing TMVA training and testing trees\n";

    loader.PrepareTrainingAndTestTree(TCut(""), "SplitMode=Block:NormMode=None:!V");//Tell TMVA that the datasets are already prepared. TMVA performs no additional normalization of event populations. Our weighs was 1 in loading so we dont want to change them. Therefore: NormMode=None

    std::cout << "TMVA datasets prepared\n";

    gRandom->SetSeed(randomSeed);//ROOT/TMVA random operations used during BDT training

    std::cout << "ROOT random seed: " << randomSeed << "\n";

    {//Create the TMVA output file and Factory
        std::cout << "Creating TMVA output file: " << tmvaOutputFile << "\n";

        std::unique_ptr<TFile> output(TFile::Open(tmvaOutputFile.c_str(), "RECREATE"));

        if (!output || output->IsZombie())
            throw std::runtime_error("Cannot create TMVA output: " + tmvaOutputFile);//TMVA's training output is stored in this ROOT file

        std::cout << "Creating TMVA Factory\n";

        TMVA::Factory factory(
            factoryName.c_str(),
            output.get(),
            "!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification"
        );//The Factory is TMVA's high-level training controller

        const std::string options = //Build the BDT configuration string
            "!H:!V"
            ":NTrees=" + std::to_string(numberOfTrees) +
            ":MaxDepth=" + std::to_string(maximumDepth) +
            ":MinNodeSize=" + std::to_string(minimumNodeSizePercent) + "%" +
            ":BoostType=Grad"
            ":Shrinkage=" + std::to_string(shrinkage) +
            ":UseBaggedBoost"
            ":BaggedSampleFraction=" + std::to_string(baggedSampleFraction) +
            ":SeparationType=GiniIndex"
            ":nCuts=" + std::to_string(numberOfCuts);

        std::cout << "BDTG configuration:"
                  << " NTrees=" << numberOfTrees
                  << " MaxDepth=" << maximumDepth
                  << " MinNodeSize=" << minimumNodeSizePercent << "%"
                  << " Shrinkage=" << shrinkage
                  << " BaggedSampleFraction=" << baggedSampleFraction
                  << " nCuts=" << numberOfCuts << "\n";

        std::cout << "Booking TMVA BDT method\n";

        if (!factory.BookMethod(&loader, TMVA::Types::kBDT, methodName.c_str(), options.c_str()))
            throw std::runtime_error("Cannot create TMVA BDT method");//Register the BDT with TMVA

        std::cout << "TMVA method booked successfully\n";
        std::cout << "Enabling TMVA multithreading with " << numberOfThreads << " threads\n";

        TMVA::gConfig().EnableMT(numberOfThreads);

        try {
            std::cout << "TMVA training START\n";
            factory.TrainAllMethods();
            factory.TestAllMethods();
            factory.EvaluateAllMethods();
            std::cout << "TMVA training END\n";
        } catch (...) {
            std::cout << "TMVA training failed, disabling TMVA multithreading\n";
            TMVA::gConfig().DisableMT();
            throw;
        }

        std::cout << "Disabling TMVA multithreading\n";
        TMVA::gConfig().DisableMT();

        std::cout << "Retrieving trained BDT method\n";

        auto* method = dynamic_cast<TMVA::MethodBDT*>(
            factory.GetMethod(datasetName.c_str(), methodName.c_str())
        );//Retrieve the trained BDT object

        if (!method)
            throw std::runtime_error("Cannot retrieve trained TMVA BDT method");

        saveBdtHistory(*method, historyRootFile);

        std::cout << "Trained BDT method retrieved successfully\n";
        std::cout << "Writing TMVA output ROOT file\n";

        output->cd();
        output->Write();//Write TMVA output

        std::cout << "TMVA output ROOT file written\n";
    }

    const std::string weightsFile =
        "MlTemp/" + datasetName + "/weights/" +
        factoryName + "_" + methodName + ".weights.xml";//Save weights in the file.

    std::cout << "Expected weights file: " << weightsFile << "\n";
    std::cout << "Checking that model weights were created\n";

    if (!std::ifstream(weightsFile))
        throw std::runtime_error("TMVA model was not created: " + weightsFile);

    std::cout << "Model weights found successfully\n";
    std::cout << "trainDrellYanBdt END\n";

    return weightsFile;
}




struct BdtPredictionResult {
    double truePositive = 0.0, trueNegative = 0.0, falsePositive = 0.0, falseNegative = 0.0;
    ULong64_t evaluatedEvents = 0;
    std::vector<std::pair<float, int>> scoredEvents;
    std::shared_ptr<std::unordered_set<ULong64_t>> selectedIndices = std::make_shared<std::unordered_set<ULong64_t>>();
    std::vector<ULong64_t> rejectedIndices;
};

BdtPredictionResult RunMcDrellYanBdtPrediction(
    const std::vector<std::string>& features, ROOT::RDF::RNode evaluationDf,
    const std::string& weightsFile, const std::string& methodName, double decisionThreshold)
{
    std::cout << "Creating lazy Take actions for evaluation features\n";

    std::vector<ROOT::RDF::RResultPtr<std::vector<Float_t>>> columns;
    columns.reserve(features.size());

    for (const auto& feature : features) columns.emplace_back(evaluationDf.Take<Float_t>(feature));

    auto labels = evaluationDf.Take<bool>("IsDyProcess");
    auto indices = evaluationDf.Take<ULong64_t>("Index");

    std::vector<ROOT::RDF::RResultHandle> handles;
    handles.reserve(columns.size() + 2);

    for (auto& column : columns) handles.emplace_back(column);
    handles.emplace_back(labels);
    handles.emplace_back(indices);

    ROOT::RDF::RunGraphs(handles);

    std::cout << "Evaluation rows loaded: " << labels->size() << "\n";
    std::cout << "Creating TMVA Reader\n";

    std::vector<Float_t> featureValues(features.size(), 0.0f);
    TMVA::Reader reader("!Color:!Silent");

    for (size_t i = 0; i < features.size(); ++i)
        reader.AddVariable(features[i].c_str(), &featureValues[i]);

    std::cout << "Loading model from: " << weightsFile << "\n";

    if (!reader.BookMVA(methodName.c_str(), weightsFile.c_str()))
        throw std::runtime_error("Cannot load TMVA model: " + weightsFile);

    std::cout << "Model loaded\n";
    std::cout << "Starting event-by-event model evaluation\n";

    BdtPredictionResult result;

    for (size_t entry = 0; entry < labels->size(); ++entry) {
        bool valid = true;

        for (size_t i = 0; i < features.size(); ++i) {
            featureValues[i] = columns[i]->at(entry);
            valid &= std::isfinite(featureValues[i]);
        }

        if (!valid) continue;

        const bool isDyProcess = labels->at(entry);
        const Float_t dyScore = reader.EvaluateMVA(methodName.c_str());

        if (!std::isfinite(dyScore)) continue;

        const bool predictedDy = dyScore >= decisionThreshold;

        if ( isDyProcess &&  predictedDy) ++result.truePositive;
        else if (!isDyProcess && !predictedDy) ++result.trueNegative;
        else if (!isDyProcess &&  predictedDy) ++result.falsePositive;
        else ++result.falseNegative;

        if (predictedDy) result.selectedIndices->insert(indices->at(entry));
        else result.rejectedIndices.push_back(indices->at(entry));

        result.scoredEvents.emplace_back(dyScore, isDyProcess ? 1 : 0);
        ++result.evaluatedEvents;
    }

    std::cout << "Event-by-event evaluation completed\n";
    return result;
}

void SaveBdtHistogram(TH1D& histogram, const std::string& file, bool logY = false)
{
    std::cout << "Saving histogram: " << file << "\n";

    TCanvas canvas(("canvas_" + std::string(histogram.GetName())).c_str(), "", 1000, 800);
    canvas.SetLeftMargin(0.14);
    canvas.SetBottomMargin(0.13);
    canvas.SetTicks(1, 1);

    if (logY) {
        canvas.SetLogy();
        histogram.SetMinimum(0.5);
    }

    histogram.SetStats(false);
    histogram.SetLineWidth(2);
    histogram.Draw("HIST");
    canvas.SaveAs(file.c_str());

    logValue("Evaluation histogram saved", file);
}

void CreateBdtPredictionHistograms(
    ROOT::RDF::RNode histogramDf,
    const std::shared_ptr<std::unordered_set<ULong64_t>>& selectedIndices)
{
    logLine("");
    logLine("--- CREATE HISTOGRAMS ---");

    auto predictedDf = histogramDf
        .Filter([selectedIndices](ULong64_t index) {
            return selectedIndices->find(index) != selectedIndices->end();
        }, {"Index"})
        .Filter([](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("EvaluationMuonKinematics",
            [](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
               const ROOT::VecOps::RVec<float>& phi) {
                int lead = -1, sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                }

                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead], pt[sub], eta[sub], phi[sub]
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("LeadingMuon_pt", "EvaluationMuonKinematics[0]")
        .Define("LeadingMuon_eta", "EvaluationMuonKinematics[1]")
        .Define("LeadingMuon_phi", "EvaluationMuonKinematics[2]")
        .Define("SubleadingMuon_pt", "EvaluationMuonKinematics[3]")
        .Define("SubleadingMuon_eta", "EvaluationMuonKinematics[4]")
        .Define("SubleadingMuon_phi", "EvaluationMuonKinematics[5]")
        .Define("EvaluationDimuonKinematics",
            [](float pt1, float eta1, float phi1, float pt2, float eta2, float phi2) {
                ROOT::Math::PtEtaPhiMVector a(pt1, eta1, phi1, 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt2, eta2, phi2, 0.105658);
                const auto d = a + b;

                return ROOT::VecOps::RVec<float>{
                    static_cast<float>(d.M()), static_cast<float>(d.Pt()),
                    static_cast<float>(d.Rapidity()), static_cast<float>(d.Phi())
                };
            }, {"LeadingMuon_pt", "LeadingMuon_eta", "LeadingMuon_phi",
                "SubleadingMuon_pt", "SubleadingMuon_eta", "SubleadingMuon_phi"})
        .Define("Dimuon_mass", "EvaluationDimuonKinematics[0]")
        .Define("Dimuon_pt", "EvaluationDimuonKinematics[1]")
        .Define("Dimuon_rapidity", "EvaluationDimuonKinematics[2]")
        .Define("Dimuon_phi", "EvaluationDimuonKinematics[3]");

    auto histogramEventCount = predictedDf.Count();

    auto hLeadingPt  = predictedDf.Histo1D({"bdt_LeadingMuon_pt", "BDT predicted DY;Leading muon p_{T} [GeV];Events", 100, 0, 140}, "LeadingMuon_pt");
    auto hLeadingEta = predictedDf.Histo1D({"bdt_LeadingMuon_eta", "BDT predicted DY;Leading muon #eta;Events", 100, -2.4, 2.4}, "LeadingMuon_eta");
    auto hLeadingPhi = predictedDf.Histo1D({"bdt_LeadingMuon_phi", "BDT predicted DY;Leading muon #phi [rad];Events", 100, -3.2, 3.2}, "LeadingMuon_phi");

    auto hSubPt  = predictedDf.Histo1D({"bdt_SubleadingMuon_pt", "BDT predicted DY;Subleading muon p_{T} [GeV];Events", 100, 0, 140}, "SubleadingMuon_pt");
    auto hSubEta = predictedDf.Histo1D({"bdt_SubleadingMuon_eta", "BDT predicted DY;Subleading muon #eta;Events", 100, -2.4, 2.4}, "SubleadingMuon_eta");
    auto hSubPhi = predictedDf.Histo1D({"bdt_SubleadingMuon_phi", "BDT predicted DY;Subleading muon #phi [rad];Events", 100, -3.2, 3.2}, "SubleadingMuon_phi");

    auto hMass = predictedDf.Histo1D({"bdt_Dimuon_mass", "BDT predicted DY;Dimuon mass [GeV];Events", 100, 50, 150}, "Dimuon_mass");
    auto hPt   = predictedDf.Histo1D({"bdt_Dimuon_pt", "BDT predicted DY;Dimuon p_{T} [GeV];Events", 100, 0, 200}, "Dimuon_pt");
    auto hRap  = predictedDf.Histo1D({"bdt_Dimuon_rapidity", "BDT predicted DY;Dimuon rapidity;Events", 100, -3, 3}, "Dimuon_rapidity");
    auto hPhi  = predictedDf.Histo1D({"bdt_Dimuon_phi", "BDT predicted DY;Dimuon #phi [rad];Events", 100, -3.2, 3.2}, "Dimuon_phi");

    std::vector<ROOT::RDF::RResultHandle> handles = {
        histogramEventCount, hLeadingPt, hLeadingEta, hLeadingPhi,
        hSubPt, hSubEta, hSubPhi, hMass, hPt, hRap, hPhi
    };

    ROOT::RDF::RunGraphs(handles);

    std::cout << "Events used for histograms: " << *histogramEventCount << "\n";
    logValue("Events used for histograms", *histogramEventCount);

    gSystem->mkdir("Images", true);

    SaveBdtHistogram(*hLeadingPt,  "Images/bdt_predicted_LeadingMuon_pt.png");
    SaveBdtHistogram(*hLeadingEta, "Images/bdt_predicted_LeadingMuon_eta.png");
    SaveBdtHistogram(*hLeadingPhi, "Images/bdt_predicted_LeadingMuon_phi.png");
    SaveBdtHistogram(*hSubPt,      "Images/bdt_predicted_SubleadingMuon_pt.png");
    SaveBdtHistogram(*hSubEta,     "Images/bdt_predicted_SubleadingMuon_eta.png");
    SaveBdtHistogram(*hSubPhi,     "Images/bdt_predicted_SubleadingMuon_phi.png");
    SaveBdtHistogram(*hMass,       "Images/bdt_predicted_Dimuon_mass.png");
    SaveBdtHistogram(*hMass,       "Images/bdt_predicted_Dimuon_mass_log.png", true);
    SaveBdtHistogram(*hPt,         "Images/bdt_predicted_Dimuon_pt.png");
    SaveBdtHistogram(*hRap,        "Images/bdt_predicted_Dimuon_rapidity.png");
    SaveBdtHistogram(*hPhi,        "Images/bdt_predicted_Dimuon_phi.png");
}

void CreateBdtDifferenceHistograms(
    ROOT::RDF::RNode histogramDf,
    const std::shared_ptr<std::unordered_set<ULong64_t>>& selectedIndices)
{
    logLine("");
    logLine("--- CREATE MODEL DIFFERENCE HISTOGRAMS ---");

    auto modelDiffDf = histogramDf
        .Filter([](bool isDy) { return isDy; }, {"IsDyProcess"})
        .Filter([selectedIndices](ULong64_t index) {
            return selectedIndices->find(index) == selectedIndices->end();
        }, {"Index"})
        .Filter([](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("ModelDiffMuonKinematics",
            [](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
               const ROOT::VecOps::RVec<float>& phi) {
                int lead = -1, sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                }

                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead], pt[sub], eta[sub], phi[sub]
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("ModelDiff_LeadingMuon_pt", "ModelDiffMuonKinematics[0]")
        .Define("ModelDiff_LeadingMuon_eta", "ModelDiffMuonKinematics[1]")
        .Define("ModelDiff_LeadingMuon_phi", "ModelDiffMuonKinematics[2]")
        .Define("ModelDiff_SubleadingMuon_pt", "ModelDiffMuonKinematics[3]")
        .Define("ModelDiff_SubleadingMuon_eta", "ModelDiffMuonKinematics[4]")
        .Define("ModelDiff_SubleadingMuon_phi", "ModelDiffMuonKinematics[5]")
        .Define("ModelDiffDimuonKinematics",
            [](float pt1, float eta1, float phi1, float pt2, float eta2, float phi2) {
                ROOT::Math::PtEtaPhiMVector a(pt1, eta1, phi1, 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt2, eta2, phi2, 0.105658);
                const auto d = a + b;

                return ROOT::VecOps::RVec<float>{
                    static_cast<float>(d.M()), static_cast<float>(d.Pt()),
                    static_cast<float>(d.Rapidity()), static_cast<float>(d.Phi())
                };
            }, {"ModelDiff_LeadingMuon_pt", "ModelDiff_LeadingMuon_eta", "ModelDiff_LeadingMuon_phi",
                "ModelDiff_SubleadingMuon_pt", "ModelDiff_SubleadingMuon_eta", "ModelDiff_SubleadingMuon_phi"})
        .Define("ModelDiff_Dimuon_mass", "ModelDiffDimuonKinematics[0]")
        .Define("ModelDiff_Dimuon_pt", "ModelDiffDimuonKinematics[1]")
        .Define("ModelDiff_Dimuon_rapidity", "ModelDiffDimuonKinematics[2]")
        .Define("ModelDiff_Dimuon_phi", "ModelDiffDimuonKinematics[3]");

    auto modelDiffCount = modelDiffDf.Count();

    auto hLeadingPt  = modelDiffDf.Histo1D({"model_diff_LeadingMuon_pt", "Hard filter DY missed by BDT;Leading muon p_{T} [GeV];Events", 100, 0, 140}, "ModelDiff_LeadingMuon_pt");
    auto hLeadingEta = modelDiffDf.Histo1D({"model_diff_LeadingMuon_eta", "Hard filter DY missed by BDT;Leading muon #eta;Events", 100, -2.4, 2.4}, "ModelDiff_LeadingMuon_eta");
    auto hLeadingPhi = modelDiffDf.Histo1D({"model_diff_LeadingMuon_phi", "Hard filter DY missed by BDT;Leading muon #phi [rad];Events", 100, -3.2, 3.2}, "ModelDiff_LeadingMuon_phi");

    auto hSubPt  = modelDiffDf.Histo1D({"model_diff_SubleadingMuon_pt", "Hard filter DY missed by BDT;Subleading muon p_{T} [GeV];Events", 100, 0, 140}, "ModelDiff_SubleadingMuon_pt");
    auto hSubEta = modelDiffDf.Histo1D({"model_diff_SubleadingMuon_eta", "Hard filter DY missed by BDT;Subleading muon #eta;Events", 100, -2.4, 2.4}, "ModelDiff_SubleadingMuon_eta");
    auto hSubPhi = modelDiffDf.Histo1D({"model_diff_SubleadingMuon_phi", "Hard filter DY missed by BDT;Subleading muon #phi [rad];Events", 100, -3.2, 3.2}, "ModelDiff_SubleadingMuon_phi");

    auto hMass = modelDiffDf.Histo1D({"model_diff_Dimuon_mass", "Hard filter DY missed by BDT;Dimuon mass [GeV];Events", 100, 50, 150}, "ModelDiff_Dimuon_mass");
    auto hPt   = modelDiffDf.Histo1D({"model_diff_Dimuon_pt", "Hard filter DY missed by BDT;Dimuon p_{T} [GeV];Events", 100, 0, 200}, "ModelDiff_Dimuon_pt");
    auto hRap  = modelDiffDf.Histo1D({"model_diff_Dimuon_rapidity", "Hard filter DY missed by BDT;Dimuon rapidity;Events", 100, -3, 3}, "ModelDiff_Dimuon_rapidity");
    auto hPhi  = modelDiffDf.Histo1D({"model_diff_Dimuon_phi", "Hard filter DY missed by BDT;Dimuon #phi [rad];Events", 100, -3.2, 3.2}, "ModelDiff_Dimuon_phi");

    std::vector<ROOT::RDF::RResultHandle> handles = {
        modelDiffCount, hLeadingPt, hLeadingEta, hLeadingPhi,
        hSubPt, hSubEta, hSubPhi, hMass, hPt, hRap, hPhi
    };

    ROOT::RDF::RunGraphs(handles);

    std::cout << "Hard-filter DY events missed by model: " << *modelDiffCount << "\n";
    logValue("Hard-filter DY events missed by model", *modelDiffCount);

    SaveBdtHistogram(*hLeadingPt,  "Images/model_diff_LeadingMuon_pt.png");
    SaveBdtHistogram(*hLeadingEta, "Images/model_diff_LeadingMuon_eta.png");
    SaveBdtHistogram(*hLeadingPhi, "Images/model_diff_LeadingMuon_phi.png");
    SaveBdtHistogram(*hSubPt,      "Images/model_diff_SubleadingMuon_pt.png");
    SaveBdtHistogram(*hSubEta,     "Images/model_diff_SubleadingMuon_eta.png");
    SaveBdtHistogram(*hSubPhi,     "Images/model_diff_SubleadingMuon_phi.png");
    SaveBdtHistogram(*hMass,       "Images/model_diff_Dimuon_mass.png");
    SaveBdtHistogram(*hPt,         "Images/model_diff_Dimuon_pt.png");
    SaveBdtHistogram(*hRap,        "Images/model_diff_Dimuon_rapidity.png");
    SaveBdtHistogram(*hPhi,        "Images/model_diff_Dimuon_phi.png");
}

void CreateBdtFalseHistograms(
    ROOT::RDF::RNode evaluationDf, std::vector<ULong64_t> rejectedIndices,
    double trueNegative, double falseNegative, double falseHistogramUsagePercentage, unsigned randomSeed)
{
    const size_t falseSampleSize = static_cast<size_t>(
        std::llround((falseHistogramUsagePercentage / 100.0) * rejectedIndices.size()));

    std::sort(rejectedIndices.begin(), rejectedIndices.end());

    std::mt19937 falseGenerator(randomSeed);
    std::shuffle(rejectedIndices.begin(), rejectedIndices.end(), falseGenerator);
    rejectedIndices.resize(falseSampleSize);

    auto falseIndices = std::make_shared<std::unordered_set<ULong64_t>>(
        rejectedIndices.begin(), rejectedIndices.end());

    logLine("");
    logLine("--- CREATE MODEL FALSE HISTOGRAMS ---");
    logValue("Non-DY histogram usage [%]", falseHistogramUsagePercentage);
    logValue("Evaluated non-DY predictions", trueNegative + falseNegative);
    logValue("Sampled non-DY predictions before muon requirements", falseSampleSize);

    auto modelFalseDf = evaluationDf
        .Filter([falseIndices](ULong64_t index) {
            return falseIndices->find(index) != falseIndices->end();
        }, {"Index"})
        .Filter([](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("ModelFalseMuonKinematics",
            [](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
               const ROOT::VecOps::RVec<float>& phi) {
                int lead = -1, sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                }

                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead], pt[sub], eta[sub], phi[sub]
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("ModelFalse_LeadingMuon_pt", "ModelFalseMuonKinematics[0]")
        .Define("ModelFalse_LeadingMuon_eta", "ModelFalseMuonKinematics[1]")
        .Define("ModelFalse_LeadingMuon_phi", "ModelFalseMuonKinematics[2]")
        .Define("ModelFalse_SubleadingMuon_pt", "ModelFalseMuonKinematics[3]")
        .Define("ModelFalse_SubleadingMuon_eta", "ModelFalseMuonKinematics[4]")
        .Define("ModelFalse_SubleadingMuon_phi", "ModelFalseMuonKinematics[5]")
        .Define("ModelFalseDimuonKinematics",
            [](float pt1, float eta1, float phi1, float pt2, float eta2, float phi2) {
                ROOT::Math::PtEtaPhiMVector a(pt1, eta1, phi1, 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt2, eta2, phi2, 0.105658);
                const auto d = a + b;

                return ROOT::VecOps::RVec<float>{
                    static_cast<float>(d.M()), static_cast<float>(d.Pt()),
                    static_cast<float>(d.Rapidity()), static_cast<float>(d.Phi())
                };
            }, {"ModelFalse_LeadingMuon_pt", "ModelFalse_LeadingMuon_eta", "ModelFalse_LeadingMuon_phi",
                "ModelFalse_SubleadingMuon_pt", "ModelFalse_SubleadingMuon_eta", "ModelFalse_SubleadingMuon_phi"})
        .Define("ModelFalse_Dimuon_mass", "ModelFalseDimuonKinematics[0]")
        .Define("ModelFalse_Dimuon_pt", "ModelFalseDimuonKinematics[1]")
        .Define("ModelFalse_Dimuon_rapidity", "ModelFalseDimuonKinematics[2]")
        .Define("ModelFalse_Dimuon_phi", "ModelFalseDimuonKinematics[3]");

    auto modelFalseCount = modelFalseDf.Count();

    auto hLeadingPt  = modelFalseDf.Histo1D({"model_false_LeadingMuon_pt", "BDT predicted non-DY;Leading muon p_{T} [GeV];Events", 100, 0, 140}, "ModelFalse_LeadingMuon_pt");
    auto hLeadingEta = modelFalseDf.Histo1D({"model_false_LeadingMuon_eta", "BDT predicted non-DY;Leading muon #eta;Events", 100, -2.4, 2.4}, "ModelFalse_LeadingMuon_eta");
    auto hLeadingPhi = modelFalseDf.Histo1D({"model_false_LeadingMuon_phi", "BDT predicted non-DY;Leading muon #phi [rad];Events", 100, -3.2, 3.2}, "ModelFalse_LeadingMuon_phi");

    auto hSubPt  = modelFalseDf.Histo1D({"model_false_SubleadingMuon_pt", "BDT predicted non-DY;Subleading muon p_{T} [GeV];Events", 100, 0, 140}, "ModelFalse_SubleadingMuon_pt");
    auto hSubEta = modelFalseDf.Histo1D({"model_false_SubleadingMuon_eta", "BDT predicted non-DY;Subleading muon #eta;Events", 100, -2.4, 2.4}, "ModelFalse_SubleadingMuon_eta");
    auto hSubPhi = modelFalseDf.Histo1D({"model_false_SubleadingMuon_phi", "BDT predicted non-DY;Subleading muon #phi [rad];Events", 100, -3.2, 3.2}, "ModelFalse_SubleadingMuon_phi");

    auto hMass = modelFalseDf.Histo1D({"model_false_Dimuon_mass", "BDT predicted non-DY;Dimuon mass [GeV];Events", 100, 50, 150}, "ModelFalse_Dimuon_mass");
    auto hPt   = modelFalseDf.Histo1D({"model_false_Dimuon_pt", "BDT predicted non-DY;Dimuon p_{T} [GeV];Events", 100, 0, 200}, "ModelFalse_Dimuon_pt");
    auto hRap  = modelFalseDf.Histo1D({"model_false_Dimuon_rapidity", "BDT predicted non-DY;Dimuon rapidity;Events", 100, -3, 3}, "ModelFalse_Dimuon_rapidity");
    auto hPhi  = modelFalseDf.Histo1D({"model_false_Dimuon_phi", "BDT predicted non-DY;Dimuon #phi [rad];Events", 100, -3.2, 3.2}, "ModelFalse_Dimuon_phi");

    std::vector<ROOT::RDF::RResultHandle> handles = {
        modelFalseCount, hLeadingPt, hLeadingEta, hLeadingPhi,
        hSubPt, hSubEta, hSubPhi, hMass, hPt, hRap, hPhi
    };

    ROOT::RDF::RunGraphs(handles);

    std::cout << "Sampled non-DY events with at least two muons: " << *modelFalseCount << "\n";
    logValue("Sampled non-DY events with at least two muons", *modelFalseCount);

    SaveBdtHistogram(*hLeadingPt,  "Images/model_false_LeadingMuon_pt.png");
    SaveBdtHistogram(*hLeadingEta, "Images/model_false_LeadingMuon_eta.png");
    SaveBdtHistogram(*hLeadingPhi, "Images/model_false_LeadingMuon_phi.png");
    SaveBdtHistogram(*hSubPt,      "Images/model_false_SubleadingMuon_pt.png");
    SaveBdtHistogram(*hSubEta,     "Images/model_false_SubleadingMuon_eta.png");
    SaveBdtHistogram(*hSubPhi,     "Images/model_false_SubleadingMuon_phi.png");
    SaveBdtHistogram(*hMass,       "Images/model_false_Dimuon_mass.png");
    SaveBdtHistogram(*hPt,         "Images/model_false_Dimuon_pt.png");
    SaveBdtHistogram(*hRap,        "Images/model_false_Dimuon_rapidity.png");
    SaveBdtHistogram(*hPhi,        "Images/model_false_Dimuon_phi.png");
}

void CreateEvaluationDataHistograms(ROOT::RDF::RNode testingDf)
{
    ROOT::RDF::RNode unfilteredDf = testingDf
        .Filter([](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("UnfilteredKinematics",
            [](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
               const ROOT::VecOps::RVec<float>& phi) {
                size_t lead = 0, sub = 1;

                if (pt[sub] > pt[lead]) std::swap(lead, sub);

                for (size_t i = 2; i < pt.size(); ++i) {
                    if (pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (pt[i] > pt[sub]) sub = i;
                }

                ROOT::Math::PtEtaPhiMVector a(pt[lead], eta[lead], phi[lead], 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt[sub], eta[sub], phi[sub], 0.105658);
                const auto d = a + b;

                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead], pt[sub], eta[sub], phi[sub],
                    static_cast<float>(d.M()), static_cast<float>(d.Pt()),
                    static_cast<float>(d.Rapidity()), static_cast<float>(d.Phi())
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"});

    struct InputHistogram {
        const char* name;
        const char* axis;
        double minimum, maximum;
    };

    const InputHistogram inputHistograms[] = {
        {"LeadingMuon_pt",     "Leading muon p_{T} [GeV]",     0, 140},
        {"LeadingMuon_eta",    "Leading muon #eta",         -2.4, 2.4},
        {"LeadingMuon_phi",    "Leading muon #phi [rad]",   -3.2, 3.2},
        {"SubleadingMuon_pt",  "Subleading muon p_{T} [GeV]",  0, 140},
        {"SubleadingMuon_eta", "Subleading muon #eta",      -2.4, 2.4},
        {"SubleadingMuon_phi", "Subleading muon #phi [rad]",-3.2, 3.2},
        {"Dimuon_mass",        "Dimuon mass [GeV]",            50, 150},
        {"Dimuon_pt",          "Dimuon p_{T} [GeV]",            0, 200},
        {"Dimuon_rapidity",    "Dimuon rapidity",              -3, 3},
        {"Dimuon_phi",         "Dimuon #phi [rad]",          -3.2, 3.2}
    };

    auto inputCount = testingDf.Count();
    auto inputHistogramCount = unfilteredDf.Count();

    std::vector<ROOT::RDF::RResultPtr<TH1D>> inputHists;
    std::vector<ROOT::RDF::RResultHandle> handles{inputCount, inputHistogramCount};

    for (size_t i = 0; i < 10; ++i) {
        const auto& spec = inputHistograms[i];
        const std::string name = "unfiltered_" + std::string(spec.name);
        const std::string title = "Input before model selection;" + std::string(spec.axis) + ";Events";

        unfilteredDf = unfilteredDf.Define(name, "UnfilteredKinematics[" + std::to_string(i) + "]");

        inputHists.emplace_back(
            unfilteredDf.Histo1D({name.c_str(), title.c_str(), 100, spec.minimum, spec.maximum}, name));

        handles.emplace_back(inputHists.back());
    }

    ROOT::RDF::RunGraphs(handles);

    std::cout << "Events passed into evaluation: " << *inputCount << '\n';
    std::cout << "Input events with at least two muons: " << *inputHistogramCount << '\n';

    logValue("Events passed into evaluation", *inputCount);
    logValue("Input events with at least two muons", *inputHistogramCount);

    gSystem->mkdir("Images", true);

    for (size_t i = 0; i < inputHists.size(); ++i)
        SaveBdtHistogram(*inputHists[i], "Images/unfiltered_" + std::string(inputHistograms[i].name) + ".png");

    SaveBdtHistogram(*inputHists[6], "Images/unfiltered_Dimuon_mass_log.png", true);
}

void CreateBackgroundPredictionHistograms(
    ROOT::RDF::RNode testingDf,
    const std::shared_ptr<std::unordered_set<ULong64_t>>& selectedIndices,
    const std::vector<std::string>& backgroundFilesPaths)
{
    struct HistSpec { const char* name; const char* axis; double min, max; };

    const HistSpec specs[] = {
        {"LeadingMuon_pt",     "Leading muon p_{T} [GeV]",      0, 140},
        {"LeadingMuon_eta",    "Leading muon #eta",          -2.4, 2.4},
        {"LeadingMuon_phi",    "Leading muon #phi [rad]",    -3.2, 3.2},
        {"SubleadingMuon_pt",  "Subleading muon p_{T} [GeV]",   0, 140},
        {"SubleadingMuon_eta", "Subleading muon #eta",       -2.4, 2.4},
        {"SubleadingMuon_phi", "Subleading muon #phi [rad]", -3.2, 3.2},
        {"Dimuon_mass",        "Dimuon mass [GeV]",             50, 150},
        {"Dimuon_pt",          "Dimuon p_{T} [GeV]",             0, 200},
        {"Dimuon_rapidity",    "Dimuon rapidity",               -3, 3},
        {"Dimuon_phi",         "Dimuon #phi [rad]",           -3.2, 3.2}
    };

    std::vector<std::string> names;
    std::vector<ROOT::RDF::RResultPtr<ULong64_t>> counts;
    std::vector<std::vector<ROOT::RDF::RResultPtr<TH1D>>> allHists;
    std::vector<ROOT::RDF::RResultHandle> handles;

    for (size_t bg = 0; bg < backgroundFilesPaths.size(); ++bg) {
        std::string name = backgroundFilesPaths[bg];
        const size_t slash = name.find_last_of("/\\");
        if (slash != std::string::npos) name = name.substr(slash + 1);

        const std::string suffix = "_xrootd_files.txt";
        if (name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0)
            name.erase(name.size() - suffix.size());

        const UInt_t fileClass = static_cast<UInt_t>(bg + 2);

        auto df = testingDf
            .Filter([fileClass](UInt_t value) { return value == fileClass; }, {"FILE_CLASSIFICATOR"})
            .Filter([selectedIndices](ULong64_t index) { return selectedIndices->find(index) != selectedIndices->end(); }, {"Index"})
            .Filter([](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                       const ROOT::VecOps::RVec<float>& phi) {
                return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
            }, {"Muon_pt", "Muon_eta", "Muon_phi"})
            .Define("BackgroundModelTrueKinematics",
                [](const ROOT::VecOps::RVec<float>& pt, const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
                    int lead = -1, sub = -1;

                    for (size_t i = 0; i < pt.size(); ++i) {
                        if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                        else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                    }

                    ROOT::Math::PtEtaPhiMVector a(pt[lead], eta[lead], phi[lead], 0.105658);
                    ROOT::Math::PtEtaPhiMVector b(pt[sub], eta[sub], phi[sub], 0.105658);
                    const auto d = a + b;

                    return ROOT::VecOps::RVec<float>{
                        pt[lead], eta[lead], phi[lead], pt[sub], eta[sub], phi[sub],
                        static_cast<float>(d.M()), static_cast<float>(d.Pt()),
                        static_cast<float>(d.Rapidity()), static_cast<float>(d.Phi())
                    };
                }, {"Muon_pt", "Muon_eta", "Muon_phi"});

        auto count = df.Count();
        handles.emplace_back(count);
        counts.emplace_back(count);

        std::vector<ROOT::RDF::RResultPtr<TH1D>> hists;

        for (size_t i = 0; i < 10; ++i) {
            const std::string column = name + "_model_true_" + specs[i].name;
            df = df.Define(column, "BackgroundModelTrueKinematics[" + std::to_string(i) + "]");

            const std::string title = name + " model predicted DY;" + specs[i].axis + ";Events";
            hists.emplace_back(df.Histo1D({column.c_str(), title.c_str(), 100, specs[i].min, specs[i].max}, column));
            handles.emplace_back(hists.back());
        }

        names.emplace_back(name);
        allHists.emplace_back(std::move(hists));
    }

    ROOT::RDF::RunGraphs(handles);
    gSystem->mkdir("Images", true);

    for (size_t bg = 0; bg < names.size(); ++bg) {
        std::cout << names[bg] << " background predicted as DY: " << *counts[bg] << "\n";
        logValue(names[bg] + " background predicted as DY", *counts[bg]);

        for (size_t i = 0; i < 10; ++i)
            SaveBdtHistogram(*allHists[bg][i], "Images/" + names[bg] + "_model_true_" + specs[i].name + ".png");

        SaveBdtHistogram(*allHists[bg][6], "Images/" + names[bg] + "_model_true_Dimuon_mass_log.png", true);
    }
}

static void CreateBdtSourceStackHistograms(
    ROOT::RDF::RNode testingDf,
    const std::shared_ptr<std::unordered_set<ULong64_t>>& indices,
    const std::vector<std::string>& backgroundFilesPaths,
    const std::string& prefix,
    const std::string& title)
{
    struct HistSpec { const char* name; const char* axis; double min, max; };

    const HistSpec specs[] = {
        {"LeadingMuon_pt",     "Leading muon p_{T} [GeV]",       0, 140},
        {"LeadingMuon_eta",    "Leading muon #eta",           -2.4, 2.4},
        {"LeadingMuon_phi",    "Leading muon #phi [rad]",     -3.2, 3.2},
        {"SubleadingMuon_pt",  "Subleading muon p_{T} [GeV]",    0, 140},
        {"SubleadingMuon_eta", "Subleading muon #eta",        -2.4, 2.4},
        {"SubleadingMuon_phi", "Subleading muon #phi [rad]",  -3.2, 3.2},
        {"Dimuon_mass",        "Dimuon mass [GeV]",              50, 150},
        {"Dimuon_pt",          "Dimuon p_{T} [GeV]",               0, 200},
        {"Dimuon_rapidity",    "Dimuon rapidity",                 -3, 3},
        {"Dimuon_phi",         "Dimuon #phi [rad]",             -3.2, 3.2}
    };

    std::vector<std::string> sourceNames{"main_dy"};

    for (std::string path : backgroundFilesPaths) {
        const size_t slash = path.find_last_of("/\\");
        if (slash != std::string::npos) path = path.substr(slash + 1);

        const std::string suffix = "_xrootd_files.txt";
        if (path.size() >= suffix.size() &&
            path.compare(path.size() - suffix.size(), suffix.size(), suffix) == 0)
            path.erase(path.size() - suffix.size());

        sourceNames.push_back(path);
    }

    const std::vector<int> colors = {
        kAzure - 9, kOrange - 3, kGreen + 2, kSpring - 6, kViolet - 6,
        kMagenta - 7, kRed - 7, kCyan - 6, kGray + 1
    };

    auto df = testingDf
        .Filter([indices](ULong64_t index) {
            return indices->find(index) != indices->end();
        }, {"Index"})
        .Filter([](const ROOT::VecOps::RVec<float>& pt,
                   const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("BdtSourceKinematics",
            [](const ROOT::VecOps::RVec<float>& pt,
               const ROOT::VecOps::RVec<float>& eta,
               const ROOT::VecOps::RVec<float>& phi) {
                int lead = -1, sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                }

                ROOT::Math::PtEtaPhiMVector a(pt[lead], eta[lead], phi[lead], 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt[sub], eta[sub], phi[sub], 0.105658);
                const auto d = a + b;

                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead], pt[sub], eta[sub], phi[sub],
                    static_cast<float>(d.M()), static_cast<float>(d.Pt()),
                    static_cast<float>(d.Rapidity()), static_cast<float>(d.Phi())
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"});

    for (size_t i = 0; i < 10; ++i)
        df = df.Define("BdtSource_" + std::string(specs[i].name),
                       "BdtSourceKinematics[" + std::to_string(i) + "]");

    std::vector<std::vector<ROOT::RDF::RResultPtr<TH1D>>> hists(sourceNames.size());
    std::vector<ROOT::RDF::RResultPtr<ULong64_t>> counts;
    std::vector<ROOT::RDF::RResultHandle> handles;

    for (size_t source = 0; source < sourceNames.size(); ++source) {
        const UInt_t fileClass = static_cast<UInt_t>(source + 1);

        auto sourceDf = df.Filter(
            [fileClass](UInt_t value) { return value == fileClass; },
            {"FILE_CLASSIFICATOR"});

        counts.emplace_back(sourceDf.Count());
        handles.emplace_back(counts.back());

        for (size_t i = 0; i < 10; ++i) {
            const std::string column = "BdtSource_" + std::string(specs[i].name);
            const std::string histName = prefix + "_" + std::to_string(source) + "_" + specs[i].name;

            hists[source].emplace_back(sourceDf.Histo1D(
                {histName.c_str(), "", 100, specs[i].min, specs[i].max}, column));

            handles.emplace_back(hists[source].back());
        }
    }

    ROOT::RDF::RunGraphs(handles);
    gSystem->mkdir("Images", true);

    for (size_t source = 0; source < sourceNames.size(); ++source) {
        std::cout << title << " - " << sourceNames[source] << ": " << *counts[source] << '\n';
        logValue(title + " - " + sourceNames[source], *counts[source]);
    }

    auto save = [&](size_t i, bool logY = false) {
        const std::string canvasName = prefix + "_canvas_" + specs[i].name;
        TCanvas canvas(canvasName.c_str(), "", 1000, 800);
        canvas.SetLeftMargin(0.14);
        canvas.SetRightMargin(0.22);
        canvas.SetBottomMargin(0.13);
        canvas.SetTicks(1, 1);
        if (logY) canvas.SetLogy();

        const std::string stackName = prefix + "_stack_" + specs[i].name;
        const std::string stackTitle = title + ";" + specs[i].axis + ";Events";
        THStack stack(stackName.c_str(), stackTitle.c_str());
        TLegend legend(0.79, 0.52, 0.98, 0.90);
        legend.SetBorderSize(0);
        legend.SetFillStyle(0);

        for (size_t source = 0; source < sourceNames.size(); ++source) {
            TH1D* hist = hists[source][i].GetPtr();
            hist->SetFillColor(colors[source % colors.size()]);
            hist->SetLineColor(kBlack);
            hist->SetLineWidth(1);
            hist->SetStats(false);
            stack.Add(hist);
            legend.AddEntry(hist, sourceNames[source].c_str(), "f");
        }

        stack.Draw("HIST");
        if (logY) stack.SetMinimum(0.5);
        stack.GetXaxis()->SetTitle(specs[i].axis);
        stack.GetYaxis()->SetTitle("Events");
        legend.Draw();

        const std::string file = "Images/" + prefix + "_" + specs[i].name +
                                 (logY ? "_log.png" : ".png");

        canvas.SaveAs(file.c_str());
        logValue("Source stacked histogram saved", file);
    };

    for (size_t i = 0; i < 10; ++i) save(i);
    save(6, true);
}


void CreateBdtPositiveSourceHistograms(
    ROOT::RDF::RNode testingDf,
    const std::shared_ptr<std::unordered_set<ULong64_t>>& selectedIndices,
    const std::vector<std::string>& backgroundFilesPaths)
{
    CreateBdtSourceStackHistograms(
        testingDf, selectedIndices, backgroundFilesPaths,
        "bdt_positive_sources", "BDT predicted DY by source");
}


void CreateBdtRejectedSourceHistograms(
    ROOT::RDF::RNode testingDf,
    const std::vector<ULong64_t>& rejectedIndices,
    const std::vector<std::string>& backgroundFilesPaths)
{
    auto rejectedSet = std::make_shared<std::unordered_set<ULong64_t>>(
        rejectedIndices.begin(), rejectedIndices.end());

    CreateBdtSourceStackHistograms(
        testingDf, rejectedSet, backgroundFilesPaths,
        "bdt_rejected_sources", "BDT rejected as non-DY by source");
}

struct RealDataBdtPrediction {
    ULong64_t index;
    Float_t score;
    bool predictedDy;
};

std::vector<RealDataBdtPrediction> RunRealDataDrellYanBdtPrediction(
    const std::vector<std::string>& features, ROOT::RDF::RNode realDataDf,
    const std::string& weightsFile, const std::string& methodName, double decisionThreshold)
{
    if (features.empty()) throw std::runtime_error("Real-data BDT feature list is empty");
    if (!realDataDf.HasColumn("Index")) throw std::runtime_error("Real-data dataframe is missing Index");

    std::string featureExpression = "ROOT::VecOps::RVec<float>{";
    for (size_t i = 0; i < features.size(); ++i) {
        if (!realDataDf.HasColumn(features[i]))
            throw std::runtime_error("Missing real-data feature: " + features[i]);
        if (i > 0) featureExpression += ", ";
        featureExpression += features[i];
    }
    featureExpression += "}";

    std::vector<Float_t> featureValues(features.size(), 0.f);
    TMVA::Reader reader("!Color:!Silent");
    for (size_t i = 0; i < features.size(); ++i)
        reader.AddVariable(features[i].c_str(), &featureValues[i]);

    if (!reader.BookMVA(methodName.c_str(), weightsFile.c_str()))
        throw std::runtime_error("Cannot load TMVA model: " + weightsFile);

    using InputRow = std::pair<ULong64_t, ROOT::VecOps::RVec<float>>;
    auto predictionDf = realDataDf
        .Define("RealDataBdtFeatures", featureExpression)
        .Define("RealDataBdtRow", [](ULong64_t index, const ROOT::VecOps::RVec<float>& values) {
            return InputRow{index, values};
        }, {"Index", "RealDataBdtFeatures"});

    auto rows = predictionDf.Take<InputRow>("RealDataBdtRow");
    const auto& inputRows = *rows; // Executes the lazy graph once.

    std::vector<RealDataBdtPrediction> result;
    result.reserve(inputRows.size());

    for (const auto& row : inputRows) {
        bool valid = true;
        for (size_t i = 0; i < features.size(); ++i) {
            featureValues[i] = row.second[i];
            valid &= std::isfinite(featureValues[i]);
        }
        if (!valid) continue;

        const Float_t score = reader.EvaluateMVA(methodName.c_str());
        if (!std::isfinite(score)) continue;
        result.push_back({row.first, score, score >= decisionThreshold});
    }

    return result;
}

void evaluateDrellYanBdt(
    const std::vector<std::string>& features,
    ROOT::RDF::RNode testingDf,
    ROOT::RDF::RNode realDataDf,
    const std::string& weightsFile,
    const std::string& methodName,
    double decisionThreshold,
    double evaluationUsagePercentage,
    double histogramUsagePercentage,
    double falseHistogramUsagePercentage,
    unsigned randomSeed,
    const std::vector<std::string>& backgroundFilesPaths)
{
    std::cout << "evaluateDrellYanBdt Start\n";
    std::cout << "Features: " << features.size() << "\n";
    std::cout << "Weights file: " << weightsFile << "\n";
    std::cout << "Method: " << methodName << "\n";
    std::cout << "Decision threshold: " << decisionThreshold << "\n";
    std::cout << "Evaluation usage: " << evaluationUsagePercentage << "%\n";
    std::cout << "Histogram usage: " << histogramUsagePercentage << "%\n";

    if (evaluationUsagePercentage <= 0.0 || evaluationUsagePercentage > 100.0)
        throw std::runtime_error("Evaluation usage percentage must be in (0, 100]");

    if (histogramUsagePercentage <= 0.0 || histogramUsagePercentage > 100.0)
        throw std::runtime_error("Histogram usage percentage must be in (0, 100]");

    if (falseHistogramUsagePercentage < 0.0 || falseHistogramUsagePercentage > 100.0)
        throw std::runtime_error("Non-DY histogram usage percentage must be in [0, 100]");


    auto testingCount = testingDf.Count();
    auto testingDyCount = testingDf.Filter([](bool isDy) { return isDy; }, {"IsDyProcess"}).Count();

    std::vector<ROOT::RDF::RResultHandle> countHandles = {testingCount, testingDyCount};
    ROOT::RDF::RunGraphs(countHandles);

    const double dyPercentage = *testingCount > 0 ? 100.0 * static_cast<double>(*testingDyCount) / static_cast<double>(*testingCount) : 0.0;

    std::cout << "Testing dataframe events: " << *testingCount << "\n";
    std::cout << "Testing dataframe DY events: " << *testingDyCount << "\n";
    std::cout << "DY events in testing dataframe: " << dyPercentage << "%\n";

    logValue("Testing dataframe events", *testingCount);
    logValue("Testing dataframe DY events", *testingDyCount);
    logValue("Testing dataframe DY [%]", dyPercentage);


    const ULong64_t keepThreshold =
        static_cast<ULong64_t>(std::llround(evaluationUsagePercentage * 100.0));

    auto evaluationDf = testingDf.Filter(
        [keepThreshold, randomSeed](ULong64_t index) {
            return (index * 7919ULL + randomSeed) % 10000ULL < keepThreshold;
        }, {"Index"});

    for (const auto& feature : features)
        if (!evaluationDf.HasColumn(feature))
            throw std::runtime_error("Missing testing feature: " + feature);

    if (!evaluationDf.HasColumn("IsDyProcess"))
        throw std::runtime_error("Testing dataframe is missing IsDyProcess");

    BdtPredictionResult prediction = RunMcDrellYanBdtPrediction(features, evaluationDf, weightsFile, methodName, decisionThreshold);

    std::cout << "Evaluated events: " << prediction.evaluatedEvents << "\n";
    std::cout << "Model-selected DY events: " << prediction.selectedIndices->size() << "\n";
    std::cout << "TP=" << prediction.truePositive << " TN=" << prediction.trueNegative << " FP=" << prediction.falsePositive << " FN=" << prediction.falseNegative << "\n";

    const double total = prediction.truePositive + prediction.trueNegative + prediction.falsePositive + prediction.falseNegative;
    const double accuracy = total > 0.0 ? (prediction.truePositive + prediction.trueNegative) / total : 0.0;
    const double precision = prediction.truePositive + prediction.falsePositive > 0.0 ? prediction.truePositive / (prediction.truePositive + prediction.falsePositive) : 0.0;
    const double recall = prediction.truePositive + prediction.falseNegative > 0.0 ? prediction.truePositive / (prediction.truePositive + prediction.falseNegative) : 0.0;
    const double specificity = prediction.trueNegative + prediction.falsePositive > 0.0 ? prediction.trueNegative / (prediction.trueNegative + prediction.falsePositive) : 0.0;
    const double f1 = precision + recall > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
    const double significance = prediction.truePositive + prediction.falsePositive > 0.0 ? prediction.truePositive / std::sqrt(prediction.truePositive + prediction.falsePositive) : 0.0;

    std::sort(prediction.scoredEvents.begin(), prediction.scoredEvents.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    double totalSignal = 0.0, totalBackground = 0.0;

    for (const auto& event : prediction.scoredEvents)
        event.second ? ++totalSignal : ++totalBackground;

    double auc = std::numeric_limits<double>::quiet_NaN();

    if (totalSignal > 0.0 && totalBackground > 0.0) {
        double cumulativeSignal = 0.0, cumulativeBackground = 0.0;
        double previousTpr = 0.0, previousFpr = 0.0, area = 0.0;

        for (size_t i = 0; i < prediction.scoredEvents.size();) {
            size_t j = i;
            double groupSignal = 0.0, groupBackground = 0.0;

            while (j < prediction.scoredEvents.size() &&
                   prediction.scoredEvents[j].first == prediction.scoredEvents[i].first) {
                prediction.scoredEvents[j].second ? ++groupSignal : ++groupBackground;
                ++j;
            }

            cumulativeSignal += groupSignal;
            cumulativeBackground += groupBackground;

            const double tpr = cumulativeSignal / totalSignal;
            const double fpr = cumulativeBackground / totalBackground;

            area += (fpr - previousFpr) * (tpr + previousTpr) * 0.5;
            previousTpr = tpr;
            previousFpr = fpr;
            i = j;
        }

        auc = area;
    }

    std::cout << "Accuracy: " << accuracy << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Background rejection: " << specificity << "\n";
    std::cout << "F1: " << f1 << "\n";
    std::cout << "ROC AUC: " << auc << "\n";
    std::cout << "S/sqrt(S+B): " << significance << "\n";

    logLine("");
    logLine("--- BDT evaluation ---");
    logValue("Evaluated events", prediction.evaluatedEvents);
    logValue("Model selected events", prediction.selectedIndices->size());
    logValue("Decision threshold", decisionThreshold);
    logValue("True positives", prediction.truePositive);
    logValue("True negatives", prediction.trueNegative);
    logValue("False positives", prediction.falsePositive);
    logValue("False negatives", prediction.falseNegative);
    logValue("Accuracy", accuracy);
    logValue("PRECISION", precision);
    logValue("Recall", recall);
    logValue("Background rejection", specificity);
    logValue("F1", f1);
    logValue("ROC AUC", auc);
    logValue("S/sqrt(S+B)", significance);

    const ULong64_t histogramKeepThreshold =
        static_cast<ULong64_t>(std::llround(histogramUsagePercentage * 100.0));

    auto histogramDf = evaluationDf.Filter(
        [histogramKeepThreshold, randomSeed](ULong64_t index) {
            return (index * 3571ULL + randomSeed) % 10000ULL < histogramKeepThreshold;
        }, {"Index"});

    // CreateBdtPredictionHistograms(histogramDf, prediction.selectedIndices);//Model predicted as true DY
    // CreateBdtFalseHistograms(evaluationDf, prediction.rejectedIndices, prediction.trueNegative, prediction.falseNegative, falseHistogramUsagePercentage, randomSeed); //Model predicted as false DY
    // CreateBdtDifferenceHistograms(histogramDf, prediction.selectedIndices);
    // CreateBackgroundPredictionHistograms(testingDf, prediction.selectedIndices, backgroundFilesPaths);
    // CreateBdtPositiveSourceHistograms(evaluationDf, prediction.selectedIndices, backgroundFilesPaths);
    // CreateBdtRejectedSourceHistograms(evaluationDf, prediction.rejectedIndices, backgroundFilesPaths);
    // CreateEvaluationDataHistograms(testingDf);

    //REAL DATA PREDICTION
    auto realDataPrediction = RunRealDataDrellYanBdtPrediction(features, realDataDf, weightsFile, methodName, decisionThreshold);

    {
        const size_t total = realDataPrediction.size();
        size_t selected = 0;
        for (const auto& event : realDataPrediction) {
            selected += event.predictedDy;
        }
        logLine("\n--- REAL-DATA BDT PREDICTION ---");
        logValue("Real-data events predicted", total);
        logValue("Real-data predicted DY", selected);
        }
    }
    

    logElapsedTime("BDT evaluation COMPLETE");
    std::cout << "evaluateDrellYanBdt END\n";
}

#endif
