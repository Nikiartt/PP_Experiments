#include <ROOT/RDataFrame.hxx>
#include <ROOT/RVec.hxx>
#include <fstream>
#include <string>
#include <map>
#include <memory>
#include <regex>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>
#include <iostream>
#include <unordered_set>
#include <iomanip>
#include <set>

static const std::string LOG_FILE = "log.txt";
using Clock = std::chrono::steady_clock;
static Clock::time_point EXECUTION_START_TIME = Clock::now();

void resetLog() { std::ofstream(LOG_FILE, std::ios::trunc) << "Drell-Yan log\n\n"; }
void resetExecutionTimer() { EXECUTION_START_TIME = Clock::now(); }
double getElapsedSeconds() { return std::chrono::duration<double>(Clock::now() - EXECUTION_START_TIME).count(); }
void logElapsedTime(const std::string& s) { std::ofstream(LOG_FILE, std::ios::app) << s << " elapsed time: " << getElapsedSeconds() << " s\n"; }
void logLine(const std::string& s) { std::ofstream(LOG_FILE, std::ios::app) << s << '\n'; }

template <typename T>
void logValue(const std::string& n, const T& v) { std::ofstream(LOG_FILE, std::ios::app) << n << ": " << v << '\n'; }


// From docker loading
// real    1m54.871s
// user    1m48.303s
// sys     0m3.923s
#if defined(__CLING__)
R__ADD_INCLUDE_PATH(/opt/libtorch/include)
R__ADD_INCLUDE_PATH(/opt/libtorch/include/torch/csrc/api/include)
R__ADD_LIBRARY_PATH(/opt/libtorch/lib)
R__LOAD_LIBRARY(/opt/libtorch/lib/libc10.so)
R__LOAD_LIBRARY(/opt/libtorch/lib/libtorch_cpu.so)
R__LOAD_LIBRARY(/opt/libtorch/lib/libtorch.so)
#endif

// After importing only separate libraries
// real    1m40.241s
// user    1m36.702s
// sys     0m3.531s
#include "ml_dy.C"

std::vector<std::string> loadFileList(const std::string& fileListPath, bool devMode, size_t devNFiles)
{
    std::ifstream input(fileListPath);
    if (!input) throw std::runtime_error("Cannot open ROOT file list: " + fileListPath);

    std::vector<std::string> files;
    std::string path;

    while (std::getline(input, path)) {
        while (!path.empty() && std::isspace(static_cast<unsigned char>(path.back())))
            path.pop_back();

        if (path.empty()) continue;

        // std::unique_ptr<TFile> testFile(TFile::Open(path.c_str(), "READ"));//Sometime s the cern storage have missing files and then we have to skip them.
        // if (!testFile || testFile->IsZombie()) { logLine("Skipping inaccessible ROOT file: " + path); continue; }

        files.push_back(path);

        if (devMode && files.size() >= devNFiles)
            break;
    }

    if (files.empty())
        throw std::runtime_error("No ROOT files found in: " + fileListPath);

    return files;
}

///FILTERING FUNCTIONS///
using ROOT::RDF::RNode;
using ROOT::VecOps::RVec;

float dphi(float a, float b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

float openingAngle3D(float eta1, float phi1, float eta2, float phi2) {
    const float c = std::clamp(
        (std::cos(dphi(phi1, phi2)) + std::sinh(eta1) * std::sinh(eta2)) /
        (std::cosh(eta1) * std::cosh(eta2)), -1.0f, 1.0f);
    return std::acos(c);
}

RNode filterHltIsoMu24(RNode df) {
    return df.Filter("HLT_IsoMu24", "HLT_IsoMu24");
}

RNode filterAtLeastTwoMuons(RNode df) {
    return df.Filter([](const RVec<float>& muPt) { return muPt.size() >= 2; },
                     {"Muon_pt"}, "At least two muons");
}

RNode filterGlobalMuons(RNode df) {
    return df.Define("GlobalMuonPairs", [](const RVec<float>& muPt, const RVec<Bool_t>& muIsGlobal) {
        RVec<int> pairs;
        for (size_t i = 0; i < muPt.size(); ++i) {
            if (!muIsGlobal[i]) continue;
            for (size_t k = i + 1; k < muPt.size(); ++k) {
                if (!muIsGlobal[k]) continue;
                pairs.push_back(static_cast<int>(i));
                pairs.push_back(static_cast<int>(k));
            }
        }
        return pairs;
    }, {"Muon_pt", "Muon_isGlobal"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"GlobalMuonPairs"}, "Global muons");
}

RNode filterMuonEta(RNode df) {
    return df.Define("EtaMuonPairs", [](const RVec<int>& pairs, const RVec<float>& muEta) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], k = pairs[p + 1];
            if (std::abs(muEta[i]) >= 2.4f || std::abs(muEta[k]) >= 2.4f) continue;
            result.push_back(i);
            result.push_back(k);
        }
        return result;
    }, {"GlobalMuonPairs", "Muon_eta"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"EtaMuonPairs"}, "Muon eta");
}

RNode filterOppositeChargeMuons(RNode df) {
    return df.Define("OppositeChargeMuonPairs", [](const RVec<int>& pairs, const RVec<int>& muCharge) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], k = pairs[p + 1];
            if (muCharge[i] * muCharge[k] >= 0) continue;
            result.push_back(i);
            result.push_back(k);
        }
        return result;
    }, {"EtaMuonPairs", "Muon_charge"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"OppositeChargeMuonPairs"}, "Opposite charge");
}

RNode filterMuonPt(RNode df) {
    return df.Define("PtMuonPairs", [](const RVec<int>& pairs, const RVec<float>& muPt) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], k = pairs[p + 1];
            const float leadPt = std::max(muPt[i], muPt[k]);
            const float subPt = std::min(muPt[i], muPt[k]);
            if (leadPt <= 22.f || subPt <= 10.f) continue;
            result.push_back(i);
            result.push_back(k);
        }
        return result;
    }, {"OppositeChargeMuonPairs", "Muon_pt"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"PtMuonPairs"}, "Muon pT");
}

RNode filterTriggerMuonId(RNode df) {
    return df.Define("MuonTriggerObjects", [](const RVec<int>& trigId) {
        RVec<int> result;
        for (size_t j = 0; j < trigId.size(); ++j)
            if (std::abs(trigId[j]) == 13) result.push_back(static_cast<int>(j));
        return result;
    }, {"TrigObj_id"})
    .Filter([](const RVec<int>& objects) { return !objects.empty(); },
            {"MuonTriggerObjects"}, "Trigger PDG ID");
}

RNode filterTriggerMuonPt(RNode df) {
    return df.Define("PtMuonTriggerObjects", [](const RVec<int>& objects, const RVec<float>& trigPt) {
        RVec<int> result;
        for (const int j : objects)
            if (trigPt[j] > 20.f) result.push_back(j);
        return result;
    }, {"MuonTriggerObjects", "TrigObj_pt"})
    .Filter([](const RVec<int>& objects) { return !objects.empty(); },
            {"PtMuonTriggerObjects"}, "Trigger pT");
}

RNode filterMuonTriggerDeltaR(RNode df) {
    return df.Filter([](const RVec<int>& pairs,
                        const RVec<float>& muEta,
                        const RVec<float>& muPhi,
                        const RVec<int>& triggerObjects,
                        const RVec<float>& trigEta,
                        const RVec<float>& trigPhi) {
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], k = pairs[p + 1];
            for (const int j : triggerObjects) {
                const float dPhi1 = dphi(muPhi[i], trigPhi[j]);
                const float dEta1 = muEta[i] - trigEta[j];
                const float dR2_1 = dEta1 * dEta1 + dPhi1 * dPhi1;

                const float dPhi2 = dphi(muPhi[k], trigPhi[j]);
                const float dEta2 = muEta[k] - trigEta[j];
                const float dR2_2 = dEta2 * dEta2 + dPhi2 * dPhi2;

                if (dR2_1 < 0.01f || dR2_2 < 0.01f) return true;
            }
        }
        return false;
    }, {"PtMuonPairs", "Muon_eta", "Muon_phi", "PtMuonTriggerObjects", "TrigObj_eta", "TrigObj_phi"},
       "Muon-trigger DeltaR");
}

RNode filterMuonStations(RNode df) {
    return df.Define("StationMuonPairs", [](const RVec<float>& pt, const RVec<int>& nStations) {
        RVec<int> pairs;
        for (size_t i = 0; i < pt.size(); ++i) {
            if (nStations[i] <= 1) continue;
            for (size_t j = i + 1; j < pt.size(); ++j) {
                if (nStations[j] <= 1) continue;
                pairs.push_back(static_cast<int>(i));
                pairs.push_back(static_cast<int>(j));
            }
        }
        return pairs;
    }, {"Muon_pt", "Muon_nStations"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"StationMuonPairs"}, "Muon stations");
}

RNode filterMuonTrackerLayers(RNode df) {
    return df.Define("TrackerLayerMuonPairs", [](const RVec<int>& pairs, const RVec<int>& nTrackerLayers) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], j = pairs[p + 1];
            if (nTrackerLayers[i] <= 5 || nTrackerLayers[j] <= 5) continue;
            result.push_back(i);
            result.push_back(j);
        }
        return result;
    }, {"StationMuonPairs", "Muon_nTrackerLayers"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"TrackerLayerMuonPairs"}, "Tracker layers");
}

RNode filterMuonDxy(RNode df) {
    return df.Define("DxyMuonPairs", [](const RVec<int>& pairs, const RVec<float>& dxy) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], j = pairs[p + 1];
            if (std::abs(dxy[i]) >= 0.2f || std::abs(dxy[j]) >= 0.2f) continue;
            result.push_back(i);
            result.push_back(j);
        }
        return result;
    }, {"TrackerLayerMuonPairs", "Muon_dxy"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"DxyMuonPairs"}, "Muon dxy");
}

RNode filterMuonPtUncertainty(RNode df) {
    return df.Define("PtErrorMuonPairs", [](const RVec<int>& pairs, const RVec<float>& pt, const RVec<float>& ptErr) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], j = pairs[p + 1];
            if (pt[i] <= 0.f || pt[j] <= 0.f) continue;
            if (ptErr[i] / pt[i] >= 0.3f || ptErr[j] / pt[j] >= 0.3f) continue;
            result.push_back(i);
            result.push_back(j);
        }
        return result;
    }, {"DxyMuonPairs", "Muon_pt", "Muon_ptErr"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"PtErrorMuonPairs"}, "Muon pT uncertainty");
}

RNode filterMuonTrackerIsolation(RNode df) {
    return df.Define("IsolatedMuonPairs", [](const RVec<int>& pairs, const RVec<float>& tkRelIso) {
        RVec<int> result;
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], j = pairs[p + 1];
            if (tkRelIso[i] >= 0.1f || tkRelIso[j] >= 0.1f) continue;
            result.push_back(i);
            result.push_back(j);
        }
        return result;
    }, {"PtErrorMuonPairs", "Muon_tkRelIso"})
    .Filter([](const RVec<int>& pairs) { return !pairs.empty(); },
            {"IsolatedMuonPairs"}, "Tracker isolation");
}

RNode filterCosmicMuons(RNode df) {
    return df.Filter([](const RVec<int>& pairs,
                        const RVec<float>& eta,
                        const RVec<float>& phi) {
        for (size_t p = 0; p < pairs.size(); p += 2) {
            const int i = pairs[p], j = pairs[p + 1];
            if (std::abs(3.14159265358979323846f -
                         openingAngle3D(eta[i], phi[i], eta[j], phi[j])) > 0.005f)
                return true;
        }
        return false;
    }, {"IsolatedMuonPairs", "Muon_eta", "Muon_phi"}, "Cosmic ray rejection");
}

//Full filter used to filter events from the real CMS data but it is not used for DY MC data set since all the events in the CM are already DY. So no sence to apply filer. It is good for filtering the jets.
//Jis tiesiog labai gerai išmeta W+Jets ir QCD foną, bet palieka beveik visą top-antitop, WW, WZ ir t.t.
RNode fullMuonFilter(RNode df) {
    df = filterHltIsoMu24(df);
    df = filterAtLeastTwoMuons(df);
    df = filterGlobalMuons(df);
    df = filterMuonEta(df);
    df = filterOppositeChargeMuons(df);
    df = filterMuonPt(df);
    df = filterTriggerMuonId(df);
    df = filterTriggerMuonPt(df);
    df = filterMuonTriggerDeltaR(df);

    df = filterMuonStations(df);
    df = filterMuonTrackerLayers(df);
    df = filterMuonDxy(df);
    df = filterMuonPtUncertainty(df);
    df = filterMuonTrackerIsolation(df);
    df = filterCosmicMuons(df);
    return df;
}

//This is more smaller filter only for DY MC data set. We leave only triger pT an n: kadangi detektoriaus geometrija ir trigeris yra dalykai, kurie paveikia visus duomenis
RNode HLT_IsoMu24_pT_n_filter(RNode df) {
    return df
        .Filter("HLT_IsoMu24", "HLT_IsoMu24")
        .Filter([](const RVec<float>& pt, const RVec<float>& eta) {
            for (size_t i = 0; i < pt.size(); ++i)
                for (size_t j = i + 1; j < pt.size(); ++j) {
                    if (std::abs(eta[i]) >= 2.4f || std::abs(eta[j]) >= 2.4f) continue;
                    const float leadPt = std::max(pt[i], pt[j]);
                    const float subPt = std::min(pt[i], pt[j]);
                    if (leadPt >= 24.f && subPt > 6.f) return true;
                }
            return false;
        }, {"Muon_pt", "Muon_eta"}, "Muon pT and eta");
}

RNode makeMuonAndDimuonKinematicsDataFrame(RNode df) {
    return df.Define("LeadingSubleadingMuonKinematics",
            [](const RVec<float>& pt, const RVec<float>& eta, const RVec<float>& phi) {
                int lead = -1, sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) {
                        sub = lead;
                        lead = i;
                    } else if (sub < 0 || pt[i] > pt[sub]) {
                        sub = i;
                    }
                }

                if (lead < 0 || sub < 0)
                    return RVec<float>{-1.f, 0.f, 0.f, -1.f, 0.f, 0.f};

                return RVec<float>{
                    pt[lead], eta[lead], phi[lead],
                    pt[sub], eta[sub], phi[sub]
                };
            },
            {"Muon_pt", "Muon_eta", "Muon_phi"}
        )
        .Define("LeadingMuon_pt", "LeadingSubleadingMuonKinematics[0]")
        .Define("LeadingMuon_eta", "LeadingSubleadingMuonKinematics[1]")
        .Define("LeadingMuon_phi", "LeadingSubleadingMuonKinematics[2]")
        .Define("SubleadingMuon_pt", "LeadingSubleadingMuonKinematics[3]")
        .Define("SubleadingMuon_eta", "LeadingSubleadingMuonKinematics[4]")
        .Define("SubleadingMuon_phi", "LeadingSubleadingMuonKinematics[5]")
        .Filter("LeadingMuon_pt > 0.f && SubleadingMuon_pt > 0.f")
        .Define("DimuonKinematics",
            [](float leadPt, float leadEta, float leadPhi,
               float subPt, float subEta, float subPhi) {
                const ROOT::Math::PtEtaPhiMVector muLead(leadPt, leadEta, leadPhi, 0.105658f);
                const ROOT::Math::PtEtaPhiMVector muSub(subPt, subEta, subPhi, 0.105658f);
                const ROOT::Math::PtEtaPhiMVector dimuon = muLead + muSub;

                return ROOT::RVecF{
                    static_cast<float>(dimuon.M()),
                    static_cast<float>(dimuon.Pt()),
                    static_cast<float>(dimuon.Rapidity()),
                    static_cast<float>(dimuon.Phi())
                };
            },
            {
                "LeadingMuon_pt",
                "LeadingMuon_eta",
                "LeadingMuon_phi",
                "SubleadingMuon_pt",
                "SubleadingMuon_eta",
                "SubleadingMuon_phi"
            }
        )
        .Define("Dimuon_mass", "DimuonKinematics[0]")
        .Define("Dimuon_pt", "DimuonKinematics[1]")
        .Define("Dimuon_rapidity", "DimuonKinematics[2]")
        .Define("Dimuon_phi", "DimuonKinematics[3]")
        .Filter("Dimuon_mass > 0.f");
}

///FILTERING FUNCTIONS END///

// Indexing Start
struct EventKey {
    UInt_t fileId;
    UInt_t run;
    UInt_t lumi;
    ULong64_t event;

    bool operator==(const EventKey& other) const noexcept {
        return fileId == other.fileId &&
               run == other.run &&
               lumi == other.lumi &&
               event == other.event;
    }
};

ULong64_t mix64(ULong64_t x)
{
    x += 0x9e3779b97f4a7c15ULL;
    x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
    x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
    return x ^ (x >> 31);
}

ULong64_t makeEventIndex(UInt_t fileId, UInt_t run, UInt_t lumi, ULong64_t event)
{
    ULong64_t h = mix64(event);
    h ^= mix64((static_cast<ULong64_t>(fileId) << 32) | run);
    h ^= mix64(lumi);
    return mix64(h);
}

struct EventKeyHash {
    std::size_t operator()(const EventKey& key) const noexcept {
        return static_cast<std::size_t>(
            makeEventIndex(key.fileId, key.run, key.lumi, key.event)
        );
    }
};
// Indexing end
struct BdtDataFrame {
    ROOT::RDF::RNode df;
    std::vector<std::string> features;
};

BdtDataFrame NormalizeToBdtDataFrame(ROOT::RDF::RNode df, size_t maxMuons)
{
    using ROOT::VecOps::RVec;

    auto node = df
        .Define("MuonOrder", [](const RVec<float>& pt) {
            RVec<size_t> order(pt.size());
            for (size_t i = 0; i < pt.size(); ++i) order[i] = i;
            std::sort(order.begin(), order.end(), [&](size_t a, size_t b) { return pt[a] > pt[b]; });
            return order;
        }, {"Muon_pt"})
        .Define("MuonCount", [](const RVec<float>& pt) { return static_cast<float>(pt.size()); }, {"Muon_pt"})
        .Define("HLT_IsoMu24_f", [](bool value) { return value ? 1.f : 0.f; }, {"HLT_IsoMu24"})
        .Define("TriggerMuonSummary", [](const RVec<int>& id, const RVec<float>& pt, const RVec<float>& eta, const RVec<float>& phi,
                                        const RVec<float>& muEta, const RVec<float>& muPhi) {
            float hasTrigger = 0.f, maxPt = 0.f, minDR2 = 100.f;
            for (size_t j = 0; j < id.size(); ++j) if (std::abs(id[j]) == 13) {
                hasTrigger = 1.f; maxPt = std::max(maxPt, pt[j]);
                for (size_t i = 0; i < muEta.size(); ++i) {
                    const float dp = dphi(muPhi[i], phi[j]), de = muEta[i] - eta[j];
                    minDR2 = std::min(minDR2, de * de + dp * dp);
                }
            }
            return RVec<float>{hasTrigger, minDR2, maxPt};
        }, {"TrigObj_id", "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "Muon_eta", "Muon_phi"})
        .Define("HasMuonTrigger", "TriggerMuonSummary[0]")
        .Define("MinMuonTriggerDeltaR2", "TriggerMuonSummary[1]")
        .Define("MaxMuonTriggerPt", "TriggerMuonSummary[2]");

    std::vector<std::string> features = {"MuonCount", "HLT_IsoMu24_f", "MET_pt", "HasMuonTrigger", "MinMuonTriggerDeltaR2", "MaxMuonTriggerPt"};

    const std::vector<std::pair<std::string, std::string>> fields = {
        {"pt", "Muon_pt"}, {"eta", "Muon_eta"}, {"charge", "Muon_charge"}, {"isGlobal", "Muon_isGlobal"},
        {"nStations", "Muon_nStations"}, {"nTrackerLayers", "Muon_nTrackerLayers"}, {"dxy", "Muon_dxy"},
        {"ptErr", "Muon_ptErr"}, {"tkRelIso", "Muon_tkRelIso"}
    };

    for (size_t i = 0; i < maxMuons; ++i) {
        const std::string exists = "Muon" + std::to_string(i) + "_exists";
        node = node.Define(exists, "MuonOrder.size() > " + std::to_string(i) + " ? 1.f : 0.f"); features.push_back(exists);

        for (const auto& [field, source] : fields) {
            const std::string name = "Muon" + std::to_string(i) + "_" + field;
            const std::string expression = "MuonOrder.size() > " + std::to_string(i) +
                " ? static_cast<float>(" + source + "[MuonOrder[" + std::to_string(i) + "]]) : 0.f";
            node = node.Define(name, expression); features.push_back(name);
        }
    }

    return {node, features};
}


std::vector<std::string> getBdtFeatureNames(size_t maxMuons) {
    std::vector<std::string> f = {"MuonCount", "HLT_IsoMu24_f", "MET_pt", "HasMuonTrigger", "MinMuonTriggerDeltaR2", "MaxMuonTriggerPt"};
    const std::vector<std::string> fields = {"exists", "pt", "eta", "charge", "isGlobal", "nStations", "nTrackerLayers", "dxy", "ptErr", "tkRelIso"};

    for (size_t i = 0; i < maxMuons; ++i)
        for (const auto& field : fields) f.push_back("Muon" + std::to_string(i) + "_" + field);

    return f;
}


//REAL DATA LUMINOSITY NORMALIZATION FOR PROPER COMPARISON START
using LumiRange = std::pair<unsigned, unsigned>;
using GoodLumiMap = std::map<unsigned, std::vector<LumiRange>>;
using RunLumiKey = std::pair<UInt_t, UInt_t>;
using LumiTable = std::map<RunLumiKey, double>;

std::vector<std::string> splitCsv(const std::string& line)
{
    std::vector<std::string> fields;
    std::stringstream stream(line);
    std::string field;
    while (std::getline(stream, field, ',')) fields.push_back(field);
    return fields;
}

UInt_t parseCsvId(const std::string& value)
{
    return static_cast<UInt_t>(std::stoul(value.substr(0, value.find(':'))));
}

ROOT::RDF::RNode filterLuminosity(ROOT::RDF::RNode df, const std::string& fileName)
{
    std::ifstream f(fileName); // Reading the full JSON file
    std::string s((std::istreambuf_iterator<char>(f)), {});//This opens the JSON file and reads the whole file into one string called s.

    auto good = std::make_shared<GoodLumiMap>();
    std::regex pairRegex(R"(\[(\d+)\s*,\s*(\d+)\])");

    size_t i = 0;
    while ((i = s.find('"', i)) != std::string::npos) {
        size_t j = s.find('"', i + 1);
        unsigned run = std::stoul(s.substr(i + 1, j - i - 1));

        i = s.find('[', j);
        size_t start = i;
        int depth = 0;

        do {
            if (s[i] == '[') ++depth;
            if (s[i] == ']') --depth;
            ++i;
        } while (depth > 0 && i < s.size());

        std::string block = s.substr(start, i - start);

        for (std::sregex_iterator it(block.begin(), block.end(), pairRegex), end; it != end; ++it) {
            unsigned first = std::stoul((*it)[1]);
            unsigned last = std::stoul((*it)[2]);
            (*good)[run].push_back({first, last});
        }
    }

    logValue("Good JSON run count", good->size()); 
    if (!good->empty()) { 
        logValue("First JSON run", good->begin()->first);
        logValue("First JSON run lumi-range count", good->begin()->second.size());
    }

    return df.Filter(
        [good](UInt_t run, UInt_t lumi) {
            GoodLumiMap::const_iterator it = good->find(run);
            if (it == good->end()) return false;

            for (const LumiRange& range : it->second) {
                if (lumi >= range.first && lumi <= range.second) return true;
            }

            return false;
        },
        {"run", "luminosityBlock"}
    );
}

LumiTable loadRecordedLuminosity(const std::string& path){
    std::ifstream input(path);
    if (!input) throw std::runtime_error("Cannot open luminosity CSV: " + path);

    LumiTable table;
    std::string line;
    size_t runCol = 0, lumiCol = 0, recordedCol = 0;
    bool headerFound = false;

    while (std::getline(input, line)) {
        if (line.rfind("#run:fill", 0) == 0) {
            const auto columns = splitCsv(line.substr(1));

            const auto columnIndex = [&columns](const std::string& name) {
                const auto it = std::find(columns.begin(), columns.end(), name);
                if (it == columns.end()) throw std::runtime_error("Missing CSV column: " + name);
                return static_cast<size_t>(std::distance(columns.begin(), it));
            };

            runCol = columnIndex("run:fill");
            lumiCol = columnIndex("ls");
            recordedCol = columnIndex("recorded(/fb)");
            headerFound = true;
            continue;
        }

        if (line.empty() || line.front() == '#') continue;
        if (!headerFound) throw std::runtime_error("Luminosity CSV header was not found.");

        const auto fields = splitCsv(line);
        if (fields.size() <= std::max({runCol, lumiCol, recordedCol})) continue;

        table[{parseCsvId(fields[runCol]), parseCsvId(fields[lumiCol])}] =
            std::stod(fields[recordedCol]);
    }

    if (table.empty()) throw std::runtime_error("No luminosity values found in: " + path);
    return table;
}

double calculateIntegratedLuminosityFb(const std::vector<std::string>& realFiles, const std::string& jsonPath, const std::string& LuminosityCsvPath){
    auto encodedKeys = filterLuminosity(ROOT::RDataFrame("LuminosityBlocks", realFiles),jsonPath)
    .Define(
        "runLumiKey",
        [](UInt_t run, UInt_t lumi) {return (static_cast<ULong64_t>(run) << 32) | lumi;},
        {"run", "luminosityBlock"}
    )
    .Take<ULong64_t>("runLumiKey");

    const std::set<ULong64_t> uniqueKeys(encodedKeys->begin(), encodedKeys->end());
    if (uniqueKeys.empty()) throw std::runtime_error("No certified luminosity blocks were found.");

    const LumiTable table = loadRecordedLuminosity(LuminosityCsvPath);
    /*
    Something like this LumiTable table looks like
    {
        {(281613, 101), 0.000001612},
        {(281613, 102), 0.000012216},
        {(284044,  30), 0.000143724}...
    }
    */
    double luminosityFb = 0.0;
    for (const ULong64_t key : uniqueKeys) {
        const RunLumiKey runLumi{
            static_cast<UInt_t>(key >> 32),
            static_cast<UInt_t>(key)
        };
        const auto it = table.find(runLumi);

        if (it == table.end()) {
            throw std::runtime_error("Missing luminosity for run " + std::to_string(runLumi.first) + ", luminosity block " + std::to_string(runLumi.second));
        }
        luminosityFb += it->second;
    }

    logValue("Unique certified luminosity blocks", uniqueKeys.size());
    logValue("Integrated luminosity [fb^-1]", luminosityFb);

    return luminosityFb;
}

double calculateNormalization(const std::string& sampleName, const std::vector<std::string>& files, double crossSectionPb, double integratedLuminosityFb)
{
    ROOT::RDataFrame runsDf("Runs", files);
    const double sumGenWeight = *runsDf.Sum<double>("genEventSumw");

    if (sumGenWeight == 0.0) {
        throw std::runtime_error("Zero genEventSumw for sample: " + sampleName);
    }

    const double scale =
        crossSectionPb * integratedLuminosityFb * 1000.0 / sumGenWeight;

    logValue(sampleName + " cross section [pb]", crossSectionPb);
    logValue(sampleName + " sum gen weights", sumGenWeight);
    logValue(sampleName + " normalization", scale);

    return scale;
}


//PYTORCH INSTALATION
// Check the environment
// uname -m
// root-config --version
// root-config --cflags

// Download and extract LibTorch
// cd /work
// curl -fL --retry 3 -o libtorch-2.6.0.zip \
//   'https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.6.0%2Bcpu.zip' &&
// python3 -m zipfile -e libtorch-2.6.0.zip /work

// Verify the files
// ls -l /work/libtorch/include/torch/csrc/api/include/torch/torch.h
// ls -l /work/libtorch/lib/{libtorch.so,libtorch_cpu.so,libc10.so}
// cat /work/libtorch/build-version

// Configure the current terminal session
// export LIBTORCH=/work/libtorch
// export ROOT_INCLUDE_PATH="$LIBTORCH/include:$LIBTORCH/include/torch/csrc/api/include${ROOT_INCLUDE_PATH:+:$ROOT_INCLUDE_PATH}"
// export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

// Test LibTorch inside ROOT
// root -l -b -q \
//   -e 'if (gSystem->Load("libc10.so") < 0 || gSystem->Load("libtorch_cpu.so") < 0 || gSystem->Load("libtorch.so") < 0) gSystem->Exit(1);' \
//   -e '#include <torch/torch.h>' \
//   -e 'std::cout << torch::ones({2, 3}) << std::endl;'

// docker exec -u root cmscloud sh -c 'mkdir -p /opt && cp -a /work/libtorch /opt/'


//REAL DATA LUMINOSITY NORMALIZATION FOR PROPER COMPARISON END

//
// docker desktop start
// docker stats cmscloud
//docker stop cmscloud OR //exit
// docker start -ai cmscloud
// root -l -b -q -e '.L ML.C' -e 'MlRun()'

// notepad $env:USERPROFILE\.wslconfig
// [wsl2]
// memory=10GB
// processors=6
// swap=4GB
// wsl --shutdown
void MlRun()
{
    std::cout << "MlRun started\n" << std::flush;
    resetExecutionTimer();
    resetLog();
    gSystem->mkdir("MlTemp", true);

    const unsigned int numberOfThreads = 4;
    // ROOT::EnableImplicitMT(numberOfThreads); //The blocks are processed concurrently, and ROOT explicitly says there is no guarantee about processing order. It also specifically warns that under implicit MT, rdfentry_ does not correspond to the original TTree/TChain entry number, and operations such as Take() do not preserve original ordering. SO DONT USE PARALELISM!

    const bool DevelopmentMode = true;
    const size_t DevelopmentFilesPerSample = 4;
    // const bool UsePreparedData = true; // false = create training/testing ROOT files, true = load them
    const double trainingFraction = 0.70;
    const std::string DyMainFilesPath = "dy_mc_main_xrootd_files.txt";// DY MONTE CARLO https://opendata.cern.ch/record/35671 
    const double DatasetUsagePercentage = 50.00;
    const double ValidationUsagePercentage = 20.00; // Percent of the selected training sample.
    const double EvaluationUsagePercentage = 100.00;
    const double HistogramUsagePercentage = 100.00;
    const double FalseHistogramUsagePercentage = 100.00;//Percentage of evaluated non-DY predictions used for model_false histograms.

    const std::string RealDataFilesPath = "files-recid-30563.txt"; //REAL CMS DATA https://opendata.cern.ch/record/30563, Goes directly into evaluation
    const size_t DevelopmentRealDataFiles = 4;
    const std::string JsonLuminosityPath = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON_MuonPhys.txt";
    const std::string LuminosityCsvPath = "Run2016H_MuonPhys_lumibyls.csv";
    
    //Parameters adjustment
    //START LOG 26/27+-
    // 2026-09-14 I set training data to 20 % and the PRECISION increased to 0.626127 (one percent acheived 68% PRECISION)
    // I increase the numberOfTrees to 40 from 30
    //increasing the  maximumDepth to 10 resi PRECISION TO 72, 15 - Precision: 0.754252
    //reduced in log32 the trees amount to 20 and the precision fall by 2-3 percents (to 0.827597)
    //log33 increased baggedSampleFraction from 0.2 to 0.4 but the percents dropped by 1.7 percent
    //log 34 reduced baggedSampleFraction 0.10 and this increased precision by 2 percents to 0.Precision: 0.843439
    //LOG 35 REDUCED THE baggedSampleFraction 0.10 to 0.05 and the precision rised to Precision: 0.857084
    //LOG 36 baggedSampleFraction = 0.01 CRASHED PRECISION TO PRECISION: 0.835005 . (So baggedSampleFraction 0.05 is the best)
    //Log 37 increased the numberOfCuts to 30 and the precision rised to Precision: 0.863429 (baggedSampleFraction was 0.05)
    //Log 38 tryed to reduce the minimumNodeSizePercent to 0.001 (from 0.005) No significant changes.
    //LOG 41 increased the training data to 10 percents and the precision dropped to 82 percents.

    const std::vector<std::string> BackgroundFilesPaths = {
        "Background_dataset_files/ttbar_xrootd_files.txt",
        "Background_dataset_files/single_top_tw_xrootd_files.txt",
        "Background_dataset_files/single_antitop_tw_xrootd_files.txt",
        "Background_dataset_files/ww_xrootd_files.txt",
        "Background_dataset_files/wz_xrootd_files.txt",
        "Background_dataset_files/zz_xrootd_files.txt",
        "Background_dataset_files/wjets_xrootd_files.txt",
        "Background_dataset_files/qcd_pt20to30_xrootd_files.txt"
    };

    const size_t MaxMuons = 2;//The data is cutted up to N muons so only 2 first muons will be slit into columns .
    const std::vector<std::string> features = getBdtFeatureNames(MaxMuons);

    //BDT PARAMETERS
    const unsigned int numberOfTrees = 6;
    const unsigned int maximumDepth = 50;//A depth-three tree can represent interactions among several variables.
    const double minimumNodeSizePercent = 0.001;//This controls how small a tree node is allowed to become.
    const double baggedSampleFraction = 0.05;//It can reduce correlations between trees and decrease overfitting. PVZ Only 20 % is used
    const unsigned int numberOfCuts = 30;//More cuts can improve split resolution but increase computation. TMVA evaluates a finite set of candidate positions
    const double shrinkage = 0.01;//It is the gradient boosting learning rate
    const unsigned int randomSeed = 1;
    const double decisionThreshold = 0.0;//decisionThreshold is the cutoff applied to the BDT output score to convert the continuous BDT prediction into a binary class.


    // NN  parameters adjustment
    // NOT IMPLEMENTED 3 PARTS DATA SPLITTING!!! // Starting point PRECISION:  0.860521,  DatasetUsagePercentage = 5.00, EvaluationUsagePercentage = 5.00, numberOfTrees = 5, nnBatchSize = 512; nnLearningRate = 0.001; nnDecisionThreshold = 0.5;, nnDropoutProbability = 0.00; nnHiddenLayers = {128, 64}; trainingFraction = 0.70, Epoch 5/5 training loss: 0.0204326 testing loss: 0.0216193, 26 inputs → Linear(128) → ReLU → Dropout(0) → Linear(64)  → ReLU → Dropout(0) → Linear(1)   → output logit.  lognn51
    // Reduce the  nnBatchSize to 256;  PRECISION: 0.837543  logNN52.txt
    // Increasing the epochs to 25  on nnBatchSize to 256; Very fluctuating PRECISION: 0.842328.
    // Decreased the epochs to 12  on nnBatchSize to 128; Very fluctuating PRECISION: from 0.882328 to final PRECISION: 0.853493. And so long time it took1700 s. logNN53.txt
    // Increase  nnBatchSize = 1024 and nnNumberOfEpochs = 7 PRECISION: 0.853008 so long nul less fluctuation. logNN54.txt

    //Returning to the inicial state of code of first NN.
    //Increase the nnNumberOfEpochs = 10; logNN55.txt PRECISION: 0.837543
    // Increasedd the nnBatchSize = 1024 PRECISION: 0.872498 logNN56.txt
    //Lets increase the train data from 5% to 10% and then increase the nnBatchSize = 2048 bad idea PRECISION: 0.846821 logNN57.txt
    // Lets add the nnDropoutProbability = 0.10 and return the data to 5 % and the nnBatchSize = 1024 PRECISION: 0.858802
    // Increase the nnDropoutProbability = 0.30 PRECISION: 0.855935 logNN58.txt
    // RETURN nnDropoutProbability = 0.00 and Increase the nngRate = 0.01 PRECISION: 0.846544 logNN59.txt
    // Reduce the nnLearningRate = 0.0005; PRECISION: 0.84724 logNN60.txt
    //Lets increase the epochs from 10 to nnNumberOfEpochs = 20 PRECISION: 0.856587 logNN61.txt
    // OK RETURNED BACK nnLearningRate = 0.001; AND 10 EPOCHS AND LETS INCREASE THE  nnWeightDecay = 0.001; PRECISION: 0.831601 logNN62.txt
    // Return back to back and created normal data local saving: nnNumberOfEpochs = 10, nnBatchSize = 1024, nnLearningRate = 0.001; nnWeightDecay = 0.0001; nnDropoutProbability = 0.00; nnHiddenLayers = {128, 64}; DATA/VALIDATION/EVALUATION - 25%/20%/20% PRECISION: 0.885293 logNN63.txt
    // nnNumberOfEpochs = 18; precision: 0.901264 logNN64.txt
    //nnNumberOfEpochs = 10 and increase nnWeightDecay = 0.001; but nothing special logNN65.txt
    // lets try the 3 layers nnHiddenLayers = {128, 128, 64}; not so resultative mut more stable logNN66.txt
    // nnHiddenLayers = {32, 64, 32}; precision: 0.903221 logNN67.txt
    // nnHiddenLayers = {256, 64}; precision: 0.892394 logNN68.txt
    // nnBatchSize = 512 Epoch 7/10 training loss: 0.0181285 testing loss: 0.0186336 testing precision: 0.896248 logNN68.txt
    //nnBatchSize = 256 precision: 0.885652 logNN70.txt
    //nnBatchSize = 32 nothing special testing precision: 0.853988 logNN71.txt
    //Increased testing to 100% nnHiddenLayers = {64, 64, 32};, nnNumberOfEpochs = 10; precision: 0.886307 epoch7 logNN72.txt
    // nnHiddenLayers = {128, 64, 32}; nnBatchSize = 512; precision: 0.886347 logNN73.txt
    // reduce nnLearningRate = 0.0007; nnBatchSize = 1024; and increase training data tp 50 % and validation same 20% evaluation 100% epoch7 precision: 0.889155 logNN74.txt
    // INCREASING THE AMOUNT OF DATA UP TO 4 FILES THEN PRECISION RISES TO PRECISION: 0.925144! (previous training on 2 filespersample) 

    //NN PARAMETERS
    const bool UseNeuralNetwork = true; //false runs the existing BDT workflow.
    const std::vector<int64_t> nnHiddenLayers = {128, 64, 32};
    const unsigned int nnNumberOfEpochs = 7;
    const int64_t nnBatchSize = 1024;
    const double nnLearningRate = 0.0007;
    const double nnWeightDecay = 0.0001;
    const double nnDropoutProbability = 0.00;
    const unsigned int nnRandomSeed = 1;
    const double nnDecisionThreshold = 0.5;
    const std::string nnModelFile = "MlTemp/DrellYan_NN_0889.pt";

    if (UseNeuralNetwork) {
        logLine("[NN model parameters]");
        logValue("Method", "LibTorch dense neural network"); logValue("Device", "CPU");
        logValue("Optimizer", "Adam"); logValue("Loss", "BCEWithLogitsLoss");
        logValue("Activation", "ReLU"); logValue("Input normalization", "Training mean and standard deviation");
        logValue("Hidden layers", nnHiddenLayers.size());
        for (size_t i = 0; i < nnHiddenLayers.size(); ++i)
            logValue("Hidden layer " + std::to_string(i + 1) + " neurons", nnHiddenLayers[i]);
        logValue("Epochs", nnNumberOfEpochs); logValue("Batch size", nnBatchSize);
        logValue("Learning rate", nnLearningRate); logValue("Weight decay", nnWeightDecay);
        logValue("Dropout probability", nnDropoutProbability); logValue("Random seed", nnRandomSeed);
        logValue("Decision threshold", nnDecisionThreshold); logValue("Model file", nnModelFile);
        logValue("Evaluation usage [%]", EvaluationUsagePercentage);
    } else {
        logLine("[BDT model parameters]");
        logValue("Method", "BDTG"); logValue("Boost type", "Gradient"); logValue("Number of trees", numberOfTrees);
        logValue("Maximum depth", maximumDepth); logValue("Minimum node size [%]", minimumNodeSizePercent);
        logValue("Bagged sample fraction", baggedSampleFraction); logValue("Number of cuts", numberOfCuts);
        logValue("Shrinkage", shrinkage); logValue("Random seed", randomSeed); logValue("Decision threshold", decisionThreshold);
    }

    const bool PrepareLocalData = false; // First run: true. Later runs: false.
    const std::string localDir = "MlTemp/NNData";

    auto percentName = [](double percent) {
        std::ostringstream s; s << percent;
        std::string name = s.str();
        std::replace(name.begin(), name.end(), '.', 'p');
        return name + "pct";
    };

    const std::string trainFile = localDir + "/DY_train_" + percentName(DatasetUsagePercentage) + ".root";
    const std::string validationFile = localDir + "/DY_validation_" + percentName(ValidationUsagePercentage) + ".root";
    const std::string evaluationFile = localDir + "/DY_evaluation_" + percentName(EvaluationUsagePercentage) + ".root";

    logLine("------");
    if (UseNeuralNetwork && !PrepareLocalData) {
        logLine("--- TRAINING AND EVALUATION USING SAVED DATA ---");
        for (const auto& file : {trainFile, validationFile, evaluationFile})
            if (gSystem->AccessPathName(file.c_str()))
                throw std::runtime_error("Local NN dataset missing: " + file);

        ROOT::DisableImplicitMT();
        ROOT::RDataFrame train("Events", trainFile);
        ROOT::RDataFrame validation("Events", validationFile);
        ROOT::RDataFrame evaluation("Events", evaluationFile);

        for (const char* column :
             {"FILE_CLASSIFICATOR", "HLT_IsoMu24", "genWeight"}) {
            if (!evaluation.HasColumn(column))
                throw std::runtime_error(
                    std::string("Saved MC evaluation file lacks ") + column +
                    "; regenerate it with PrepareLocalData = true");
        }

        // const std::string weights = trainDrellYanNn(
        //     features, train,
        //     validation,
        //     nnModelFile,
        //     nnHiddenLayers,
        //     nnNumberOfEpochs,
        //     nnBatchSize,
        //     nnLearningRate,
        //     nnWeightDecay,
        //     nnDropoutProbability,
        //     nnRandomSeed,
        //     100.0,//100%usage of reading file
        //     numberOfThreads,
        //     nnDecisionThreshold);

        //Real data loading from Online for NN evaluation
                // The saved-data NN branch returns before the BDT real-data setup below.
        const auto nnRealFiles = loadFileList(
            RealDataFilesPath, DevelopmentMode, DevelopmentRealDataFiles);
        const double nnLuminosityFb = calculateIntegratedLuminosityFb(
            nnRealFiles, JsonLuminosityPath, LuminosityCsvPath);

        std::vector<std::vector<std::string>> nnMcFilesBySource{
            loadFileList(DyMainFilesPath, DevelopmentMode, DevelopmentFilesPerSample)
        };
        for (const auto& path : BackgroundFilesPaths)
            nnMcFilesBySource.push_back(
                loadFileList(path, DevelopmentMode, DevelopmentFilesPerSample));

        const std::vector<double> nnCrossSectionsPb = {
            5396.0, 687.1, 32.45, 32.51, 75.88, 27.56, 12.08, 54110.0, 2.529e6
        };
        if (nnCrossSectionsPb.size() != nnMcFilesBySource.size())
            throw std::runtime_error("NN MC cross sections do not match the source list");

        std::vector<double> nnMcScales;
        nnMcScales.reserve(nnMcFilesBySource.size());
        for (size_t i = 0; i < nnMcFilesBySource.size(); ++i) {
            const std::string name = i == 0 ? "main_dy" : BackgroundFilesPaths[i - 1];
            nnMcScales.push_back(calculateNormalization(
                name, nnMcFilesBySource[i], nnCrossSectionsPb[i], nnLuminosityFb));
        }

        ROOT::RDataFrame nnRealData("Events", nnRealFiles);
        auto nnRealWithIndex = nnRealData
            .DefinePerSample("SourceFileId",
                [nnRealFiles](unsigned int, const ROOT::RDF::RSampleInfo& info) -> UInt_t {
                    for (UInt_t i = 0; i < nnRealFiles.size(); ++i)
                        if (info.Contains(nnRealFiles[i])) return i;
                    throw std::runtime_error("Cannot determine real-data source file");
                })
            .Define("Index",
                [](UInt_t fileId, UInt_t run, UInt_t lumi, ULong64_t event) {
                    return makeEventIndex(fileId, run, lumi, event);
                }, {"SourceFileId", "run", "luminosityBlock", "event"});

        auto nnRealForTesting = nnRealWithIndex.Filter(
            [trainingFraction](ULong64_t index) {
                return !(static_cast<double>(index % 100) / 100.0 < trainingFraction);
            }, {"Index"});

        auto nnFilteredReal = HLT_IsoMu24_pT_n_filter(
            filterLuminosity(nnRealForTesting, JsonLuminosityPath));
        auto nnRealFeatures = NormalizeToBdtDataFrame(nnFilteredReal, MaxMuons);
        if (nnRealFeatures.features != features)
            throw std::runtime_error("Real-data and NN feature lists differ");

        // The saved MC evaluation file already has this evaluation fraction applied.
        // Apply the same fraction and seed to real data.
        const ULong64_t nnRealKeep = static_cast<ULong64_t>(
            std::llround(EvaluationUsagePercentage * 100.0));
        auto nnRealEvaluation = nnRealFeatures.df.Filter(
            [nnRealKeep, nnRandomSeed](ULong64_t index) {
                return mix64(index ^ static_cast<ULong64_t>(nnRandomSeed) ^
                             0xd1b54a32d192ed03ULL) % 10000ULL < nnRealKeep;
            }, {"Index"});

        evaluateDrellYanNn(
            features,
            evaluation,
            nnRealEvaluation,
            // weights,
            "MlTemp/DrellYan_NN_0889.pt",
            nnDecisionThreshold,
            100.0, // Saved MC evaluation file was already sampled.
            HistogramUsagePercentage,
            FalseHistogramUsagePercentage,
            nnRandomSeed,
            nnBatchSize,
            numberOfThreads,
            BackgroundFilesPaths,
            nnMcScales);
        logElapsedTime("MlRun complete");
        return;
    }
    
    logLine("--- CREATE TRAINING AND TESTING DATA ---");

    const auto dyMainFiles = loadFileList(DyMainFilesPath, DevelopmentMode, DevelopmentFilesPerSample);

    std::vector<std::string> allFiles = dyMainFiles;
    std::vector<UInt_t> allFileClasses(dyMainFiles.size(), 1);
    std::vector<std::vector<std::string>> mcFilesBySource{dyMainFiles};

    for (size_t i = 0; i < BackgroundFilesPaths.size(); ++i) {
        const auto files = loadFileList(BackgroundFilesPaths[i], DevelopmentMode, DevelopmentFilesPerSample);
        mcFilesBySource.push_back(files);
        allFiles.insert(allFiles.end(), files.begin(), files.end());
        allFileClasses.insert(allFileClasses.end(), files.size(), static_cast<UInt_t>(i + 2));
    }

    ROOT::EnableImplicitMT(numberOfThreads);
    ROOT::RDataFrame dfMainAndBg("Events", allFiles);

    auto dfMarkedEvents = dfMainAndBg
        .DefinePerSample("SourceFileId",
            [allFiles](unsigned int, const ROOT::RDF::RSampleInfo& info) -> UInt_t {
                for (UInt_t i = 0; i < allFiles.size(); ++i)
                    if (info.Contains(allFiles[i])) return i;
                throw std::runtime_error("Cannot determine source file");
            })
        .Define("FILE_CLASSIFICATOR",
            [allFileClasses](UInt_t fileId) { return allFileClasses.at(fileId); },
            {"SourceFileId"});

    auto dfWithIndex = dfMarkedEvents.Define(
        "Index",
        [](UInt_t fileId, UInt_t run, UInt_t lumi, ULong64_t event) {
            return makeEventIndex(fileId, run, lumi, event);
        },
        {"SourceFileId", "run", "luminosityBlock", "event"});

    auto dfWithKey = dfWithIndex.Define(
        "EventKeyParts",
        [](UInt_t fileId, UInt_t run, UInt_t lumi, ULong64_t event) {
            return ROOT::VecOps::RVec<ULong64_t>{
                static_cast<ULong64_t>(fileId),
                static_cast<ULong64_t>(run),
                static_cast<ULong64_t>(lumi),
                event
            };
        },
        {"SourceFileId", "run", "luminosityBlock", "event"}
    );

    auto filteredDyEvents = HLT_IsoMu24_pT_n_filter(dfWithKey.Filter("FILE_CLASSIFICATOR == 1"));

    auto passedDyKeysResult = filteredDyEvents.Take<ROOT::VecOps::RVec<ULong64_t>>("EventKeyParts");

    const auto& passedRows = *passedDyKeysResult;

    auto passedDyKeys = std::make_shared<std::unordered_set<EventKey, EventKeyHash>>();

    passedDyKeys->reserve(passedRows.size() * 2);

    for (const auto& row : passedRows) {
        passedDyKeys->insert({
            static_cast<UInt_t>(row[0]),
            static_cast<UInt_t>(row[1]),
            static_cast<UInt_t>(row[2]),
            row[3]
        });
    }

    //Events that fail filter remain available in dfWithKey. So labeledDyDf contains all events from all loaded files
    auto labeledDyDf = dfWithKey
        .Define("IsDyProcess",
            [passedDyKeys](UInt_t fileId, UInt_t run, UInt_t lumi, ULong64_t event, UInt_t fileClass) {
                if (fileClass != 1) return false;
                return passedDyKeys->find({fileId, run, lumi, event}) != passedDyKeys->end();
            },
            {"SourceFileId", "run", "luminosityBlock", "event", "FILE_CLASSIFICATOR"})
        .Filter("FILE_CLASSIFICATOR != 1 || IsDyProcess");//Adiitional filter that removes the IsDyProcess = 0

    auto bdtData = NormalizeToBdtDataFrame(labeledDyDf, MaxMuons);


    std::cout << "\n--- BDT DATA NORMALIZATION ---\n";
    std::cout << "Max muons per event: " << MaxMuons << '\n';
    std::cout << "Input columns: " << labeledDyDf.GetColumnNames().size() << '\n';
    std::cout << "Normalized columns: " << bdtData.df.GetColumnNames().size() << '\n';
    std::cout << "BDT scalar features: " << features.size() << '\n';

    for (const auto& feature : features)
        std::cout << "  " << feature << " [" << bdtData.df.GetColumnType(feature) << "]\n";

    std::cout << "Normalization completed\n";

    auto dfWithSplit = bdtData.df.Define("IsTraining",
        [trainingFraction](ULong64_t index) {
            return static_cast<double>(index % 100) / 100.0 < trainingFraction;
        }, {"Index"});


    std::vector<std::string> cachedColumns = features;

    cachedColumns.emplace_back("Index");
    cachedColumns.emplace_back("IsDyProcess");
    cachedColumns.emplace_back("IsTraining");

    auto cachedBdtData = dfWithSplit.Cache(cachedColumns);

    ROOT::RDF::RNode trainingNode = dfWithSplit.Filter("IsTraining"); //trainingNode contains 30% of all events from all loaded files
    ROOT::RDF::RNode testingNode  = dfWithSplit.Filter("!IsTraining"); //testingNode contains 70% of all events from all loaded files
    
    if (UseNeuralNetwork && PrepareLocalData) {
        if (ValidationUsagePercentage <= 0.0 || ValidationUsagePercentage >= 100.0)
            throw std::runtime_error("Validation percentage must be between 0 and 100");

        gSystem->mkdir(localDir.c_str(), true);
        const auto trainKeep = static_cast<ULong64_t>(std::llround(DatasetUsagePercentage * 100.0));
        const auto validationKeep = static_cast<ULong64_t>(std::llround(ValidationUsagePercentage * 100.0));
        const auto evalKeep = static_cast<ULong64_t>(std::llround(EvaluationUsagePercentage * 100.0));

        const auto selectTrain = [trainKeep, nnRandomSeed](ULong64_t index) {
            return (index * 7919ULL + nnRandomSeed) % 10000ULL < trainKeep;
        };
        const auto selectValidation = [validationKeep, nnRandomSeed](ULong64_t index) {
            return mix64(index ^ static_cast<ULong64_t>(nnRandomSeed) ^
                        0x243f6a8885a308d3ULL) % 10000ULL < validationKeep;
        };
        const auto selectEvaluation = [evalKeep, nnRandomSeed](ULong64_t index) {
            return mix64(index ^ static_cast<ULong64_t>(nnRandomSeed) ^
                        0xd1b54a32d192ed03ULL) % 10000ULL < evalKeep;
        };

        std::vector<std::string> columns = features;
        for (const char* name : {"Muon_pt", "Muon_eta", "Muon_phi"})
            if (std::find(columns.begin(), columns.end(), name) == columns.end())
                columns.emplace_back(name);
        columns.push_back("Index");
        columns.push_back("IsDyProcess");
        columns.push_back("FILE_CLASSIFICATOR");
        columns.push_back("HLT_IsoMu24");
        columns.push_back("genWeight");

        ROOT::RDF::RSnapshotOptions options;
        options.fLazy = true;
        options.fCompressionAlgorithm = ROOT::RCompressionSetting::EAlgorithm::kLZ4;
        options.fCompressionLevel = 4;

        auto selectedTraining = trainingNode.Filter(selectTrain, {"Index"});
        auto savedTrain = selectedTraining
            .Filter([selectValidation](ULong64_t index) { return !selectValidation(index); }, {"Index"})
            .Snapshot("Events", trainFile, columns, options);
        auto savedValidation = selectedTraining.Filter(selectValidation, {"Index"})
            .Snapshot("Events", validationFile, columns, options);
        auto savedEvaluation = testingNode.Filter(selectEvaluation, {"Index"})
            .Snapshot("Events", evaluationFile, columns, options);

        ROOT::RDF::RunGraphs(std::vector<ROOT::RDF::RResultHandle>{
            savedTrain, savedValidation, savedEvaluation
        });
        logElapsedTime("Local NN datasets saved");
        return;
    }

    std::cout << "cachedBdtData completed\n";

    logElapsedTime("Training/testing dataframes created");

    logLine("--- REAL DATA LOADING ---");
    const auto realDataFiles = loadFileList(RealDataFilesPath, DevelopmentMode, DevelopmentRealDataFiles);
    // Calculate luminosity from all certified blocks in the loaded real-data files.
    const double integratedLuminosityFb = calculateIntegratedLuminosityFb(realDataFiles, JsonLuminosityPath, LuminosityCsvPath);
    // Same cross sections and source order as comparisonAnalysis().
    const std::vector<double> crossSectionsPb = {5396.0, 687.1, 32.45, 32.51, 75.88, 27.56, 12.08, 54110.0, 2.529e6}; // DY, ttbar, top-tW, antitop-tW, WW, WZ, ZZ, W+jets, QCD.

    if (crossSectionsPb.size() != mcFilesBySource.size())
        throw std::runtime_error("MC cross sections do not match the source list");

    std::vector<double> mcScales;
    for (size_t i = 0; i < mcFilesBySource.size(); ++i) {
        const std::string name = i == 0 ? "main_dy" : BackgroundFilesPaths[i - 1];
        mcScales.push_back(calculateNormalization(name, mcFilesBySource[i], crossSectionsPb[i], integratedLuminosityFb));}

    // Append Indexes to the realdata dataframe
    ROOT::RDataFrame realDataDf("Events", realDataFiles);
    auto realDataWithIndex = realDataDf
        .DefinePerSample("SourceFileId", [realDataFiles](unsigned int, const ROOT::RDF::RSampleInfo& info) -> UInt_t {
            for (UInt_t i = 0; i < realDataFiles.size(); ++i)
                if (info.Contains(realDataFiles[i])) return i;
            throw std::runtime_error("Cannot determine real-data source file");
        })
        .Define("Index", [](UInt_t fileId, UInt_t run, UInt_t lumi, ULong64_t event) {
            return makeEventIndex(fileId, run, lumi, event);
        }, {"SourceFileId", "run", "luminosityBlock", "event"});

    auto realDataForTesting = realDataWithIndex.Filter(
        [trainingFraction](ULong64_t index) {
            return !(static_cast<double>(index % 100) / 100.0 < trainingFraction);
        }, {"Index"});

    auto filteredRealData = HLT_IsoMu24_pT_n_filter(filterLuminosity(realDataForTesting, JsonLuminosityPath));
    auto realBdtData = NormalizeToBdtDataFrame(filteredRealData, MaxMuons);

    if (realBdtData.features != features)
        throw std::runtime_error("Real-data and MC BDT feature lists differ");


    // Main and BG events will all the amount of muons pass in the training.
    const std::string weightsFile = trainDrellYanBdt(
        features,
        trainingNode,
        testingNode,
        "MlTemp/TMVA_DY_training.root",
        "MlTemp/BDT_training_history.root",
        "DrellYanClassification.weights.xml",
        "BDTG",
        "DY_BDT_dataset",
        numberOfTrees,
        maximumDepth,
        minimumNodeSizePercent,
        baggedSampleFraction,
        numberOfCuts,
        shrinkage,
        randomSeed,
        DatasetUsagePercentage,
        numberOfThreads
    );
    logValue("Trained model", weightsFile); logElapsedTime("BDT training END");


    // EVALUATION
    evaluateDrellYanBdt(
        features,
        testingNode,
        realBdtData.df,
        // weightsFile,
        "MlTemp/DY_BDT_dataset/weights/DrellYanClassification.weights.xml_BDTG.weights.xml",
        "BDTG",
        decisionThreshold,
        EvaluationUsagePercentage,
        HistogramUsagePercentage,
        FalseHistogramUsagePercentage,
        randomSeed,
        BackgroundFilesPaths,
        mcScales
    );
    logElapsedTime("BDT evaluation END");


    logElapsedTime("MlRun complete");
}

