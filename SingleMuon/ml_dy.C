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

    std::cout << "All " << features.size() << " features exist in training and testing dataframes\n";

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


void evaluateDrellYanBdt(
    const std::vector<std::string>& features,
    ROOT::RDF::RNode testingDf,
    const std::string& weightsFile,
    const std::string& methodName,
    double decisionThreshold,
    double evaluationUsagePercentage,
    double histogramUsagePercentage,
    unsigned randomSeed)
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

    const ULong64_t keepThreshold =
        static_cast<ULong64_t>(std::llround(evaluationUsagePercentage * 100.0));

    std::cout << "Evaluation keep threshold: " << keepThreshold << "/10000\n";
    std::cout << "Applying evaluation dataset percentage filter\n";

    auto evaluationDf = testingDf.Filter(
        [keepThreshold, randomSeed](ULong64_t index) {
            return (index * 7919ULL + randomSeed) % 10000ULL < keepThreshold;
        }, {"Index"});

    std::cout << "Cutted evaluationDf\n";
    std::cout << "Checking evaluation features\n";

    for (const auto& feature : features)
        if (!evaluationDf.HasColumn(feature))
            throw std::runtime_error("Missing testing feature: " + feature);

    std::cout << "All " << features.size() << " evaluation features exist\n";

    if (!evaluationDf.HasColumn("IsDyProcess"))
        throw std::runtime_error("Testing dataframe is missing IsDyProcess");

    std::cout << "IsDyProcess label column exists\n";
    std::cout << "Creating lazy Take actions for evaluation features\n";

    std::vector<ROOT::RDF::RResultPtr<std::vector<Float_t>>> columns;
    columns.reserve(features.size());

    for (const auto& feature : features)
        columns.emplace_back(evaluationDf.Take<Float_t>(feature));

    std::cout << "Creating lazy Take actions for labels and indices\n";

    auto labels = evaluationDf.Take<bool>("IsDyProcess");
    auto indices = evaluationDf.Take<ULong64_t>("Index");

    std::vector<ROOT::RDF::RResultHandle> handles;
    handles.reserve(columns.size() + 2);

    for (auto& column : columns) handles.emplace_back(column);
    handles.emplace_back(labels);
    handles.emplace_back(indices);

    std::cout << "Prepared " << handles.size() << " evaluation RDataFrame actions\n";
    std::cout << "Executing evaluation RDataFrame graphs\n";

    ROOT::RDF::RunGraphs(handles);

    std::cout << "Evaluation RDataFrame graphs completed\n";
    std::cout << "Evaluation rows loaded: " << labels->size() << "\n";
    std::cout << "Creating TMVA Reader\n";

    std::vector<Float_t> featureValues(features.size(), 0.0f);
    TMVA::Reader reader("!Color:!Silent");

    std::cout << "Registering TMVA Reader variables\n";

    for (size_t i = 0; i < features.size(); ++i)
        reader.AddVariable(features[i].c_str(), &featureValues[i]);

    std::cout << "Registered " << features.size() << " Reader variables\n";
    std::cout << "Loading model from: " << weightsFile << "\n";

    if (!reader.BookMVA(methodName.c_str(), weightsFile.c_str()))
        throw std::runtime_error("Cannot load TMVA model: " + weightsFile);

    std::cout << "Model loaded\n";
    std::cout << "Starting event-by-event model evaluation\n";

    double truePositive = 0.0, trueNegative = 0.0, falsePositive = 0.0, falseNegative = 0.0;
    ULong64_t evaluatedEvents = 0;
    std::vector<std::pair<float, int>> scoredEvents;
    auto selectedIndices = std::make_shared<std::unordered_set<ULong64_t>>();

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

        if      ( isDyProcess &&  predictedDy) ++truePositive;
        else if (!isDyProcess && !predictedDy) ++trueNegative;
        else if (!isDyProcess &&  predictedDy) ++falsePositive;
        else                                    ++falseNegative;

        if (predictedDy) selectedIndices->insert(indices->at(entry));

        scoredEvents.emplace_back(dyScore, isDyProcess ? 1 : 0);
        ++evaluatedEvents;
    }

    std::cout << "Event-by-event evaluation completed\n";
    std::cout << "Evaluated events: " << evaluatedEvents << "\n";
    std::cout << "Model-selected DY events: " << selectedIndices->size() << "\n";
    std::cout << "TP=" << truePositive << " TN=" << trueNegative
              << " FP=" << falsePositive << " FN=" << falseNegative << "\n";

    std::cout << "Calculating evaluation metrics\n";

    const double total = truePositive + trueNegative + falsePositive + falseNegative;
    const double accuracy = total > 0.0 ? (truePositive + trueNegative) / total : 0.0;
    const double precision = truePositive + falsePositive > 0.0 ? truePositive / (truePositive + falsePositive) : 0.0;
    const double recall = truePositive + falseNegative > 0.0 ? truePositive / (truePositive + falseNegative) : 0.0;
    const double specificity = trueNegative + falsePositive > 0.0 ? trueNegative / (trueNegative + falsePositive) : 0.0;
    const double f1 = precision + recall > 0.0 ? 2.0 * precision * recall / (precision + recall) : 0.0;
    const double significance = truePositive + falsePositive > 0.0 ? truePositive / std::sqrt(truePositive + falsePositive) : 0.0;

    std::cout << "Accuracy: " << accuracy << "\n";
    std::cout << "Precision: " << precision << "\n";
    std::cout << "Recall: " << recall << "\n";
    std::cout << "Background rejection: " << specificity << "\n";
    std::cout << "F1: " << f1 << "\n";
    std::cout << "S/sqrt(S+B): " << significance << "\n";
    std::cout << "Sorting model scores for ROC AUC\n";

    std::sort(scoredEvents.begin(), scoredEvents.end(),
        [](const auto& a, const auto& b) { return a.first > b.first; });

    double totalSignal = 0.0, totalBackground = 0.0;

    for (const auto& event : scoredEvents)
        event.second ? ++totalSignal : ++totalBackground;

    std::cout << "ROC events - signal: " << totalSignal
              << ", background: " << totalBackground << "\n";

    double auc = std::numeric_limits<double>::quiet_NaN();

    if (totalSignal > 0.0 && totalBackground > 0.0) {
        std::cout << "Calculating ROC AUC\n";

        double cumulativeSignal = 0.0, cumulativeBackground = 0.0;
        double previousTpr = 0.0, previousFpr = 0.0, area = 0.0;

        for (size_t i = 0; i < scoredEvents.size();) {
            size_t j = i;
            double groupSignal = 0.0, groupBackground = 0.0;

            while (j < scoredEvents.size() && scoredEvents[j].first == scoredEvents[i].first) {
                scoredEvents[j].second ? ++groupSignal : ++groupBackground;
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

    std::cout << "Calculated parameters\n";
    std::cout << "ROC AUC: " << auc << "\n";
    std::cout << "Writing evaluation metrics to log\n";

    logLine("");
    logLine("--- BDT evaluation ---");
    logValue("Evaluated events", evaluatedEvents);
    logValue("Model selected events", selectedIndices->size());
    logValue("Decision threshold", decisionThreshold);
    logValue("True positives", truePositive);
    logValue("True negatives", trueNegative);
    logValue("False positives", falsePositive);
    logValue("False negatives", falseNegative);
    logValue("Accuracy", accuracy);
    logValue("PRECISION", precision);
    logValue("Recall", recall);
    logValue("Background rejection", specificity);
    logValue("F1", f1);
    logValue("ROC AUC", auc);
    logValue("S/sqrt(S+B)", significance);




    // HISTOGRAMS //
    logLine("");
    logLine("--- CREATE HISTOGRAMS ---");

    std::cout << "Histogram creation START\n";

    const ULong64_t histogramKeepThreshold =
        static_cast<ULong64_t>(std::llround(histogramUsagePercentage * 100.0));

    std::cout << "Histogram keep threshold: " << histogramKeepThreshold << "/10000\n";
    std::cout << "Applying histogram dataset percentage filter\n";

    auto histogramDf = evaluationDf.Filter(
        [histogramKeepThreshold, randomSeed](ULong64_t index) {
            return (index * 3571ULL + randomSeed) % 10000ULL < histogramKeepThreshold;
        }, {"Index"});

    std::cout << "Creating dataframe containing model-selected events\n";

    auto predictedDf = histogramDf
        .Filter([selectedIndices](ULong64_t index) {
            return selectedIndices->find(index) != selectedIndices->end();
        }, {"Index"})
        .Filter([](const ROOT::VecOps::RVec<float>& pt,
                   const ROOT::VecOps::RVec<float>& eta,
                   const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("EvaluationMuonKinematics",
            [](const ROOT::VecOps::RVec<float>& pt,
               const ROOT::VecOps::RVec<float>& eta,
               const ROOT::VecOps::RVec<float>& phi) {
                int lead = -1, sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                }

                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead],
                    pt[sub], eta[sub], phi[sub]
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("LeadingMuon_pt",  "EvaluationMuonKinematics[0]")
        .Define("LeadingMuon_eta", "EvaluationMuonKinematics[1]")
        .Define("LeadingMuon_phi", "EvaluationMuonKinematics[2]")
        .Define("SubleadingMuon_pt",  "EvaluationMuonKinematics[3]")
        .Define("SubleadingMuon_eta", "EvaluationMuonKinematics[4]")
        .Define("SubleadingMuon_phi", "EvaluationMuonKinematics[5]")
        .Define("EvaluationDimuonKinematics",
            [](float pt1, float eta1, float phi1,
               float pt2, float eta2, float phi2) {
                ROOT::Math::PtEtaPhiMVector a(pt1, eta1, phi1, 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt2, eta2, phi2, 0.105658);
                const auto d = a + b;

                return ROOT::VecOps::RVec<float>{
                    (float)d.M(), (float)d.Pt(), (float)d.Rapidity(), (float)d.Phi()
                };
            }, {"LeadingMuon_pt", "LeadingMuon_eta", "LeadingMuon_phi",
                "SubleadingMuon_pt", "SubleadingMuon_eta", "SubleadingMuon_phi"})
        .Define("Dimuon_mass",     "EvaluationDimuonKinematics[0]")
        .Define("Dimuon_pt",       "EvaluationDimuonKinematics[1]")
        .Define("Dimuon_rapidity", "EvaluationDimuonKinematics[2]")
        .Define("Dimuon_phi",      "EvaluationDimuonKinematics[3]");

    std::cout << "Predicted DY histogram dataframe prepared\n";
    std::cout << "Creating histogram actions\n";

    auto histogramEventCount = predictedDf.Count();

    auto hLeadingPt  = predictedDf.Histo1D({"bdt_LeadingMuon_pt",  "BDT predicted DY;Leading muon p_{T} [GeV];Events", 100, 0, 140}, "LeadingMuon_pt");
    auto hLeadingEta = predictedDf.Histo1D({"bdt_LeadingMuon_eta", "BDT predicted DY;Leading muon #eta;Events", 100, -2.4, 2.4}, "LeadingMuon_eta");
    auto hLeadingPhi = predictedDf.Histo1D({"bdt_LeadingMuon_phi", "BDT predicted DY;Leading muon #phi [rad];Events", 100, -3.2, 3.2}, "LeadingMuon_phi");

    auto hSubPt  = predictedDf.Histo1D({"bdt_SubleadingMuon_pt",  "BDT predicted DY;Subleading muon p_{T} [GeV];Events", 100, 0, 140}, "SubleadingMuon_pt");
    auto hSubEta = predictedDf.Histo1D({"bdt_SubleadingMuon_eta", "BDT predicted DY;Subleading muon #eta;Events", 100, -2.4, 2.4}, "SubleadingMuon_eta");
    auto hSubPhi = predictedDf.Histo1D({"bdt_SubleadingMuon_phi", "BDT predicted DY;Subleading muon #phi [rad];Events", 100, -3.2, 3.2}, "SubleadingMuon_phi");

    auto hMass = predictedDf.Histo1D({"bdt_Dimuon_mass", "BDT predicted DY;Dimuon mass [GeV];Events", 100, 50, 150}, "Dimuon_mass");
    auto hPt   = predictedDf.Histo1D({"bdt_Dimuon_pt", "BDT predicted DY;Dimuon p_{T} [GeV];Events", 100, 0, 200}, "Dimuon_pt");
    auto hRap  = predictedDf.Histo1D({"bdt_Dimuon_rapidity", "BDT predicted DY;Dimuon rapidity;Events", 100, -3, 3}, "Dimuon_rapidity");
    auto hPhi  = predictedDf.Histo1D({"bdt_Dimuon_phi", "BDT predicted DY;Dimuon #phi [rad];Events", 100, -3.2, 3.2}, "Dimuon_phi");

    std::vector<ROOT::RDF::RResultHandle> histogramHandles = {
        histogramEventCount,
        hLeadingPt, hLeadingEta, hLeadingPhi,
        hSubPt, hSubEta, hSubPhi,
        hMass, hPt, hRap, hPhi
    };

    std::cout << "Prepared " << histogramHandles.size() << " histogram actions\n";
    std::cout << "Executing histogram RDataFrame graphs\n";

    ROOT::RDF::RunGraphs(histogramHandles);

    std::cout << "Histogram RDataFrame graphs completed\n";
    std::cout << "Events used for histograms: " << *histogramEventCount << "\n";

    logValue("Events used for histograms", *histogramEventCount);

    std::cout << "Creating Images directory\n";
    gSystem->mkdir("Images", true);

    auto save = [](auto& hist, const std::string& file, bool logY = false) {
        std::cout << "Saving histogram: " << file << "\n";

        TCanvas canvas(("canvas_" + std::string(hist->GetName())).c_str(), "", 1000, 800);
        canvas.SetLeftMargin(0.14);
        canvas.SetBottomMargin(0.13);
        canvas.SetTicks(1, 1);

        if (logY) {
            canvas.SetLogy();
            hist->SetMinimum(0.5);
        }

        hist->SetStats(false);
        hist->SetLineWidth(2);
        hist->Draw("HIST");

        canvas.SaveAs(file.c_str());
        logValue("Evaluation histogram saved", file);
    };

    std::cout << "Saving evaluation histograms\n";

    save(hLeadingPt,  "Images/bdt_predicted_LeadingMuon_pt.png");
    save(hLeadingEta, "Images/bdt_predicted_LeadingMuon_eta.png");
    save(hLeadingPhi, "Images/bdt_predicted_LeadingMuon_phi.png");
    save(hSubPt,      "Images/bdt_predicted_SubleadingMuon_pt.png");
    save(hSubEta,     "Images/bdt_predicted_SubleadingMuon_eta.png");
    save(hSubPhi,     "Images/bdt_predicted_SubleadingMuon_phi.png");
    save(hMass,       "Images/bdt_predicted_Dimuon_mass.png");
    save(hMass,       "Images/bdt_predicted_Dimuon_mass_log.png", true);
    save(hPt,         "Images/bdt_predicted_Dimuon_pt.png");
    save(hRap,        "Images/bdt_predicted_Dimuon_rapidity.png");
    save(hPhi,        "Images/bdt_predicted_Dimuon_phi.png");

    std::cout << "All evaluation histograms saved\n";
    std::cout << "Histogram creation END\n";

    
    logLine("");
    logLine("--- CREATE MODEL DIFFERENCE HISTOGRAMS ---");

    auto modelDiffDf = histogramDf
        .Filter([](bool isDy) { return isDy; }, {"IsDyProcess"})
        .Filter([selectedIndices](ULong64_t index) {
            return selectedIndices->find(index) == selectedIndices->end();
        }, {"Index"})
        .Filter([](const ROOT::VecOps::RVec<float>& pt,
                const ROOT::VecOps::RVec<float>& eta,
                const ROOT::VecOps::RVec<float>& phi) {
            return pt.size() >= 2 && pt.size() == eta.size() && pt.size() == phi.size();
        }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("ModelDiffMuonKinematics",
            [](const ROOT::VecOps::RVec<float>& pt,
            const ROOT::VecOps::RVec<float>& eta,
            const ROOT::VecOps::RVec<float>& phi) {
                int lead = -1, sub = -1;
                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) { sub = lead; lead = i; }
                    else if (sub < 0 || pt[i] > pt[sub]) sub = i;
                }
                return ROOT::VecOps::RVec<float>{
                    pt[lead], eta[lead], phi[lead],
                    pt[sub], eta[sub], phi[sub]
                };
            }, {"Muon_pt", "Muon_eta", "Muon_phi"})
        .Define("ModelDiff_LeadingMuon_pt",  "ModelDiffMuonKinematics[0]")
        .Define("ModelDiff_LeadingMuon_eta", "ModelDiffMuonKinematics[1]")
        .Define("ModelDiff_LeadingMuon_phi", "ModelDiffMuonKinematics[2]")
        .Define("ModelDiff_SubleadingMuon_pt",  "ModelDiffMuonKinematics[3]")
        .Define("ModelDiff_SubleadingMuon_eta", "ModelDiffMuonKinematics[4]")
        .Define("ModelDiff_SubleadingMuon_phi", "ModelDiffMuonKinematics[5]")
        .Define("ModelDiffDimuonKinematics",
            [](float pt1, float eta1, float phi1,
            float pt2, float eta2, float phi2) {
                ROOT::Math::PtEtaPhiMVector a(pt1, eta1, phi1, 0.105658);
                ROOT::Math::PtEtaPhiMVector b(pt2, eta2, phi2, 0.105658);
                const auto d = a + b;
                return ROOT::VecOps::RVec<float>{
                    (float)d.M(), (float)d.Pt(),
                    (float)d.Rapidity(), (float)d.Phi()
                };
            }, {
                "ModelDiff_LeadingMuon_pt",
                "ModelDiff_LeadingMuon_eta",
                "ModelDiff_LeadingMuon_phi",
                "ModelDiff_SubleadingMuon_pt",
                "ModelDiff_SubleadingMuon_eta",
                "ModelDiff_SubleadingMuon_phi"
            })
        .Define("ModelDiff_Dimuon_mass",     "ModelDiffDimuonKinematics[0]")
        .Define("ModelDiff_Dimuon_pt",       "ModelDiffDimuonKinematics[1]")
        .Define("ModelDiff_Dimuon_rapidity", "ModelDiffDimuonKinematics[2]")
        .Define("ModelDiff_Dimuon_phi",      "ModelDiffDimuonKinematics[3]");

    auto modelDiffCount = modelDiffDf.Count();

    auto hDiffLeadingPt = modelDiffDf.Histo1D(
        {"model_diff_LeadingMuon_pt", "Hard filter DY missed by BDT;Leading muon p_{T} [GeV];Events", 100, 0, 140},
        "ModelDiff_LeadingMuon_pt");

    auto hDiffLeadingEta = modelDiffDf.Histo1D(
        {"model_diff_LeadingMuon_eta", "Hard filter DY missed by BDT;Leading muon #eta;Events", 100, -2.4, 2.4},
        "ModelDiff_LeadingMuon_eta");

    auto hDiffLeadingPhi = modelDiffDf.Histo1D(
        {"model_diff_LeadingMuon_phi", "Hard filter DY missed by BDT;Leading muon #phi [rad];Events", 100, -3.2, 3.2},
        "ModelDiff_LeadingMuon_phi");

    auto hDiffSubPt = modelDiffDf.Histo1D(
        {"model_diff_SubleadingMuon_pt", "Hard filter DY missed by BDT;Subleading muon p_{T} [GeV];Events", 100, 0, 140},
        "ModelDiff_SubleadingMuon_pt");

    auto hDiffSubEta = modelDiffDf.Histo1D(
        {"model_diff_SubleadingMuon_eta", "Hard filter DY missed by BDT;Subleading muon #eta;Events", 100, -2.4, 2.4},
        "ModelDiff_SubleadingMuon_eta");

    auto hDiffSubPhi = modelDiffDf.Histo1D(
        {"model_diff_SubleadingMuon_phi", "Hard filter DY missed by BDT;Subleading muon #phi [rad];Events", 100, -3.2, 3.2},
        "ModelDiff_SubleadingMuon_phi");

    auto hDiffMass = modelDiffDf.Histo1D(
        {"model_diff_Dimuon_mass", "Hard filter DY missed by BDT;Dimuon mass [GeV];Events", 100, 50, 150},
        "ModelDiff_Dimuon_mass");

    auto hDiffPt = modelDiffDf.Histo1D(
        {"model_diff_Dimuon_pt", "Hard filter DY missed by BDT;Dimuon p_{T} [GeV];Events", 100, 0, 200},
        "ModelDiff_Dimuon_pt");

    auto hDiffRap = modelDiffDf.Histo1D(
        {"model_diff_Dimuon_rapidity", "Hard filter DY missed by BDT;Dimuon rapidity;Events", 100, -3, 3},
        "ModelDiff_Dimuon_rapidity");

    auto hDiffPhi = modelDiffDf.Histo1D(
        {"model_diff_Dimuon_phi", "Hard filter DY missed by BDT;Dimuon #phi [rad];Events", 100, -3.2, 3.2},
        "ModelDiff_Dimuon_phi");

    std::vector<ROOT::RDF::RResultHandle> modelDiffHandles = {
        modelDiffCount,
        hDiffLeadingPt, hDiffLeadingEta, hDiffLeadingPhi,
        hDiffSubPt, hDiffSubEta, hDiffSubPhi,
        hDiffMass, hDiffPt, hDiffRap, hDiffPhi
    };

    ROOT::RDF::RunGraphs(modelDiffHandles);

    std::cout << "Hard-filter DY events missed by model: "
            << *modelDiffCount << "\n";

    logValue("Hard-filter DY events missed by model", *modelDiffCount);

    save(hDiffLeadingPt,  "Images/model_diff_LeadingMuon_pt.png");
    save(hDiffLeadingEta, "Images/model_diff_LeadingMuon_eta.png");
    save(hDiffLeadingPhi, "Images/model_diff_LeadingMuon_phi.png");

    save(hDiffSubPt,  "Images/model_diff_SubleadingMuon_pt.png");
    save(hDiffSubEta, "Images/model_diff_SubleadingMuon_eta.png");
    save(hDiffSubPhi, "Images/model_diff_SubleadingMuon_phi.png");

    save(hDiffMass, "Images/model_diff_Dimuon_mass.png");
    save(hDiffPt,   "Images/model_diff_Dimuon_pt.png");
    save(hDiffRap,  "Images/model_diff_Dimuon_rapidity.png");
    save(hDiffPhi,  "Images/model_diff_Dimuon_phi.png");




    logElapsedTime("BDT evaluation COMPLETE");

    std::cout << "evaluateDrellYanBdt END\n";
}

#endif
