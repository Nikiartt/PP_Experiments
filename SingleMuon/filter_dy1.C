// #include <ROOT/RDataFrame.hxx>
// #include <ROOT/RVec.hxx>
// #include <fstream>
// #include <string>
// #include <map>
// #include <memory>
// #include <regex>
// #include <vector>
// #include <cmath>
// #include <sstream>
// #include <chrono>
// #include <TCanvas.h>
// #include <TGraph.h>
// #include <limits>
// #include <algorithm>
// #include <stdexcept>
// #include <Math/Vector4D.h>
// #include <TH1D.h>
// #include <TLegend.h>
// #include <THStack.h>
// #include "ml_dy.C"
// #include <optional>
// #include <iostream>
// #include <unordered_set>
// #include <cctype>
// #include <iomanip>
// #include <set>
// #include <TLine.h>
// #include <TPad.h>
// #include <TChain.h>
// #include <mgl2/mgl.h>

using namespace ROOT::VecOps;

using LumiRange = std::pair<unsigned, unsigned>; //Convenient types definition
using GoodLumiMap = std::map<unsigned, std::vector<LumiRange>>;
using GoodLumiMapPtr = std::shared_ptr<GoodLumiMap>;

/////LOGS////
static const std::string LOG_FILE = "log.txt"; 

void resetLog() { 
    std::ofstream(LOG_FILE, std::ios::trunc) << "Drell-Yan log\n\n";
}

//time logs
using Clock = std::chrono::steady_clock;
static Clock::time_point EXECUTION_START_TIME = Clock::now();

void resetExecutionTimer()
{
    EXECUTION_START_TIME = Clock::now();
}

double getElapsedSeconds()
{
    return std::chrono::duration<double>(Clock::now() - EXECUTION_START_TIME).count();
}

void logElapsedTime(const std::string& label)
{
    std::ofstream(LOG_FILE, std::ios::app)
        << label << " elapsed time: " << getElapsedSeconds() << " s\n";
}

void logLine(const std::string& text) { 
    std::ofstream(LOG_FILE, std::ios::app) << text << "\n";
}

template <typename T>
void logValue(const std::string& name, const T& value) { 
    std::ofstream(LOG_FILE, std::ios::app) << name << ": " << value << "\n";
}

void logCutCount(const std::string& name, ULong64_t count) { 
    std::ofstream(LOG_FILE, std::ios::app) << name << ": " << count << "\n";
}

void logDataFrameShape(const std::string& name, ROOT::RDF::RNode df)
{
    const ULong64_t events = *df.Count();
    const size_t     cols  = df.GetColumnNames().size();

    std::ofstream(LOG_FILE, std::ios::app)
        << name << " shape: (" << events << " events, "
        << cols  << " columns)\n";
}

template <typename T>
void logFirstUniqueValues(ROOT::RDF::RNode df, const std::string& columnName, size_t maxValues = 5){
    auto values = df.Take<T>(columnName);

    std::vector<T> uniqueValues;
    bool hasMore = false;

    for (const T& value : *values) {
        bool alreadySeen = false;

        for (const T& uniqueValue : uniqueValues) {
            if (uniqueValue == value) {
                alreadySeen = true;
                break;
            }
        }

        if (alreadySeen) continue;

        if (uniqueValues.size() < maxValues) {
            uniqueValues.push_back(value);
        } else {
            hasMore = true;
            break;
        }
    }

    std::ofstream out(LOG_FILE, std::ios::app);
    out << columnName << " first unique: ";

    if (uniqueValues.empty()) {
        out << "<empty>\n";
        return;
    }

    for (size_t i = 0; i < uniqueValues.size(); ++i) {
        if (i > 0) out << " ";
        out << uniqueValues[i];
    }

    if (hasMore) out << " ...";
    out << "\n";
}

template <typename T>
void logFirstUniqueValuesInRVec(ROOT::RDF::RNode df, const std::string& columnName, size_t maxValues = 5)
{
    auto values = df.Take<RVec<T>>(columnName);

    std::vector<T> uniqueValues;
    bool hasMore = false;

    for (const RVec<T>& eventValues : *values) {
        for (const T& value : eventValues) {
            bool alreadySeen = false;

            for (const T& uniqueValue : uniqueValues) {
                if (uniqueValue == value) {
                    alreadySeen = true;
                    break;
                }
            }

            if (alreadySeen) continue;

            if (uniqueValues.size() < maxValues) {
                uniqueValues.push_back(value);
            } else {
                hasMore = true;
                break;
            }
        }

        if (hasMore) break;
    }

    std::ofstream out(LOG_FILE, std::ios::app);
    out << columnName << " first unique (RVec): ";

    if (uniqueValues.empty()) {
        out << "<empty>\n";
        return;
    }

    for (size_t i = 0; i < uniqueValues.size(); ++i) {
        if (i > 0) out << " ";
        out << uniqueValues[i];
    }

    if (hasMore) out << " ...";
    out << "\n";
}

void logFirstMatchedEvents(ROOT::RDF::RNode df, unsigned n = 5) { 
    ROOT::RDF::RNode s = df.Range(n);

    auto runs = s.Take<UInt_t>("run");
    auto lumis = s.Take<UInt_t>("luminosityBlock");
    auto events = s.Take<ULong64_t>("event");
    auto muPts = s.Take<RVec<float>>("Muon_pt");
    auto muEtas = s.Take<RVec<float>>("Muon_eta");
    auto muPhis = s.Take<RVec<float>>("Muon_phi");

    std::ofstream out(LOG_FILE, std::ios::app);
    out << "\n[First HLT matched events]\n";

    for (size_t i = 0; i < runs->size(); ++i) {
        out << "event_index: " << i << "\n";
        out << "run: " << (*runs)[i] << "\n";
        out << "luminosityBlock: " << (*lumis)[i] << "\n";
        out << "event: " << (*events)[i] << "\n";

        out << "Muon_pt: ";
        for (float x : (*muPts)[i]) out << x << " ";
        out << "\n";

        out << "Muon_eta: ";
        for (float x : (*muEtas)[i]) out << x << " ";
        out << "\n";

        out << "Muon_phi: ";
        for (float x : (*muPhis)[i]) out << x << " ";
        out << "\n\n";
    }
}

template <typename T>
void logMinMaxValues(ROOT::RDF::RNode df, const std::string& columnName)
{
    auto minValue = df.Min<T>(columnName);
    auto maxValue = df.Max<T>(columnName);

    std::ofstream out(LOG_FILE, std::ios::app);
    out << columnName << " min: " << *minValue << "\n";
    out << columnName << " max: " << *maxValue << "\n";
}

template <typename T>
void logMinMaxValuesInRVec(ROOT::RDF::RNode df, const std::string& columnName)
{
    auto values = df.Take<RVec<T>>(columnName);

    bool hasValue = false;
    T minValue = std::numeric_limits<T>::max();
    T maxValue = std::numeric_limits<T>::lowest();

    for (const RVec<T>& eventValues : *values) {
        for (const T& value : eventValues) {
            if (!hasValue) {
                minValue = value;
                maxValue = value;
                hasValue = true;
                continue;
            }

            if (value < minValue) minValue = value;
            if (value > maxValue) maxValue = value;
        }
    }

    std::ofstream out(LOG_FILE, std::ios::app);

    if (!hasValue) {
        out << columnName << " min: <empty>\n";
        out << columnName << " max: <empty>\n";
        return;
    }

    out << columnName << " min: " << minValue << "\n";
    out << columnName << " max: " << maxValue << "\n";
}

void logMuonCountMinMax(ROOT::RDF::RNode df, const std::string& muonColumn = "Muon_pt")
{
    auto counts = df.Define("nMuonsInEvent", muonColumn + ".size()");

    auto minMuons = counts.Min<ULong_t>("nMuonsInEvent");
    auto maxMuons = counts.Max<ULong_t>("nMuonsInEvent");

    logValue("Min muons in event", *minMuons);
    logValue("Max muons in event", *maxMuons);
}

/////LOGS////

///// GRAPHS /////
void applyCanvasPadding(TCanvas& canvas)
{
    canvas.SetLeftMargin(0.16);
    canvas.SetRightMargin(0.18);
    canvas.SetTopMargin(0.12);
    canvas.SetBottomMargin(0.14);
    canvas.SetTicks(1, 1);
}

void applyAxisStyle(TAxis* xAxis, TAxis* yAxis)
{
    xAxis->SetTitleOffset(1.20);
    yAxis->SetTitleOffset(1.45);

    xAxis->SetLabelSize(0.04);
    yAxis->SetLabelSize(0.04);

    xAxis->SetTitleSize(0.045);
    yAxis->SetTitleSize(0.045);
}

void applyGraphStyle(TGraph& graph)
{
    graph.SetMarkerStyle(20);
    graph.SetMarkerSize(0.4);
    applyAxisStyle(graph.GetXaxis(), graph.GetYaxis());
}

void applyHistogramStyle(TH1D& hist)
{
    hist.SetLineWidth(2);
    hist.SetMarkerStyle(0);
    applyAxisStyle(hist.GetXaxis(), hist.GetYaxis());
}

template <typename Drawable>
void savePlot(Drawable& object, const std::string& drawOption, const std::string& outputFile)
{
    TCanvas canvas("canvas", "plot", 1000, 800);
    applyCanvasPadding(canvas);

    object.Draw(drawOption.c_str());

    canvas.Modified();
    canvas.Update();
    canvas.SaveAs(outputFile.c_str());
}

void createScatterPlot(ROOT::RDF::RNode df, const std::string& xColumn, const std::string& yColumn, const std::string& outputFile)
{
    auto graph = df.Graph(xColumn, yColumn);

    graph->SetTitle((yColumn + " vs " + xColumn + ";" + xColumn + ";" + yColumn).c_str());
    applyGraphStyle(*graph);

    savePlot(*graph, "AP", outputFile);
    logValue("Scatter plot saved", outputFile);
}

template <typename X, typename Y>
void createScatterPlotFromRVec(ROOT::RDF::RNode df, const std::string& xColumn, const std::string& yColumn, const std::string& outputFile, ULong64_t maxPoints = 100000)
{
    auto xEvents = df.Take<RVec<X>>(xColumn);
    auto yEvents = df.Take<RVec<Y>>(yColumn);

    TGraph graph;

    ULong64_t point = 0;

    for (size_t e = 0; e < xEvents->size() && point < maxPoints; ++e) {
        const auto& xs = (*xEvents)[e];
        const auto& ys = (*yEvents)[e];

        for (size_t i = 0, n = std::min(xs.size(), ys.size()); i < n && point < maxPoints; ++i) {
            graph.SetPoint(point++, static_cast<double>(xs[i]), static_cast<double>(ys[i]));
        }
    }

    graph.SetTitle((yColumn + " vs " + xColumn + ";" + xColumn + ";" + yColumn).c_str());
    applyGraphStyle(graph);

    savePlot(graph, "AP", outputFile);

    logValue("Scatter plot saved", outputFile);
    logValue("Scatter plot x column", xColumn);
    logValue("Scatter plot y column", yColumn);
    logValue("Scatter plot points", point);
}

template <typename T>
void createRVecHistogram(ROOT::RDF::RNode df, const std::string& columnName, int bins, const std::string& outputFile, double xMin = std::numeric_limits<double>::quiet_NaN(), double xMax = std::numeric_limits<double>::quiet_NaN())
{
    auto values = df.Take<RVec<T>>(columnName);

    std::vector<double> flatValues;

    for (const auto& eventValues : *values) {
        for (const auto& value : eventValues) {
            flatValues.push_back(static_cast<double>(value));
        }
    }

    if (flatValues.empty()) {
        logValue("Histogram skipped empty column", columnName);
        return;
    }

    const auto [minIt, maxIt] = std::minmax_element(flatValues.begin(), flatValues.end());

    const double histMin = std::isnan(xMin) ? *minIt : xMin;
    const double histMax = std::isnan(xMax) ? *maxIt : xMax;

    if (histMin >= histMax) {
        logValue("Histogram skipped invalid range", columnName);
        return;
    }

    TH1D hist(
        ("hist_" + columnName).c_str(),
        (columnName + ";" + columnName + ";Muon count").c_str(),
        bins,
        histMin,
        histMax
    );

    for (const double value : flatValues) {
        hist.Fill(value);
    }

    applyHistogramStyle(hist);
    savePlot(hist, "HIST", outputFile);

    logValue("Histogram saved", outputFile);
}

template <typename T>
ROOT::RDF::RResultPtr<TH1D> bookRVecHistogram(ROOT::RDF::RNode df, const std::string& columnName, int bins, const std::string& title, double xMin, double xMax)
{
    return df.Histo1D<T>(
        {
            ("hist_" + columnName).c_str(),
            (title + ";" + columnName + ";Muon count").c_str(),
            bins,
            xMin,
            xMax
        },
        columnName
    );
}

void saveHistogram(ROOT::RDF::RResultPtr<TH1D> hist, const std::string& outputFile){
    TCanvas canvas("canvas", "histogram", 1200, 900);
    canvas.cd();

    hist->Draw("HIST");

    gPad->Modified();
    gPad->Update();

    canvas.SaveAs(outputFile.c_str());

    logValue("Histogram saved", outputFile);
}

void saveComparisonHistogram(ROOT::RDF::RResultPtr<TH1D> realHist, ROOT::RDF::RResultPtr<TH1D> mcHist, const std::string& outputFile, const std::string& realLabel = "Real data", const std::string& mcLabel = "MC + backgrounds")
{
    TCanvas canvas("comparisonCanvas", "comparison histogram", 1200, 1000);

    TPad upperPad("upperPad", "histograms", 0.0, 0.30, 1.0, 1.0);
    TPad ratioPad("ratioPad", "ratio", 0.0, 0.0, 1.0, 0.30);

    upperPad.SetLeftMargin(0.14); upperPad.SetRightMargin(0.05); upperPad.SetTopMargin(0.08); upperPad.SetBottomMargin(0.02);
    ratioPad.SetLeftMargin(0.14); ratioPad.SetRightMargin(0.05); ratioPad.SetTopMargin(0.03); ratioPad.SetBottomMargin(0.35);
    upperPad.Draw(); ratioPad.Draw();

    applyHistogramStyle(*realHist);
    applyHistogramStyle(*mcHist);

    realHist->SetStats(false); mcHist->SetStats(false);
    realHist->SetLineColor(kBlack); realHist->SetLineWidth(2);
    mcHist->SetLineColor(kRed + 1); mcHist->SetLineWidth(2);

    upperPad.cd();

    const double maxY = std::max(realHist->GetMaximum(), mcHist->GetMaximum());
    realHist->SetMaximum(maxY > 0.0 ? maxY * 1.25 : 1.0);
    realHist->SetMinimum(0.0);
    realHist->GetXaxis()->SetLabelSize(0.0);
    realHist->GetXaxis()->SetTitleSize(0.0);

    realHist->Draw("HIST");
    mcHist->Draw("HIST SAME");

    TLegend legend(0.67, 0.77, 0.93, 0.90);
    legend.SetBorderSize(0); legend.SetFillStyle(0);
    legend.AddEntry(realHist.GetPtr(), realLabel.c_str(), "l");
    legend.AddEntry(mcHist.GetPtr(), mcLabel.c_str(), "l");
    legend.Draw();

    ratioPad.cd();

    TH1D ratio(*mcHist.GetPtr());
    ratio.SetDirectory(nullptr);
    ratio.SetName("mc_data_ratio");
    ratio.SetTitle("");
    ratio.SetStats(false);
    ratio.Divide(realHist.GetPtr());

    double ratioMaximum = 0.0;
    for (int bin = 1; bin <= ratio.GetNbinsX(); ++bin) {
        if (realHist->GetBinContent(bin) > 0.0) ratioMaximum = std::max(ratioMaximum, ratio.GetBinContent(bin));
    }

    ratio.SetMinimum(0.0);
    ratio.SetMaximum(std::max(2.0, ratioMaximum * 1.20));
    ratio.SetLineColor(kRed + 1);
    ratio.SetLineWidth(2);

    ratio.GetYaxis()->SetTitle("MC / Data");
    ratio.GetYaxis()->SetNdivisions(505);
    ratio.GetYaxis()->SetTitleSize(0.10);
    ratio.GetYaxis()->SetTitleOffset(0.55);
    ratio.GetYaxis()->SetLabelSize(0.085);

    ratio.GetXaxis()->SetTitle(realHist->GetXaxis()->GetTitle());
    ratio.GetXaxis()->SetTitleSize(0.12);
    ratio.GetXaxis()->SetTitleOffset(1.10);
    ratio.GetXaxis()->SetLabelSize(0.095);

    ratio.Draw("HIST");

    TLine unityLine(ratio.GetXaxis()->GetXmin(), 1.0, ratio.GetXaxis()->GetXmax(), 1.0);
    unityLine.SetLineColor(kGray + 2);
    unityLine.SetLineStyle(2);
    unityLine.SetLineWidth(2);
    unityLine.Draw("SAME");

    canvas.Modified();
    canvas.Update();
    canvas.SaveAs(outputFile.c_str());

    logValue("Comparison histogram saved", outputFile);
}

struct McSample {
    int id;
    std::string name;
    double crossSectionPb;
    double normalizationScale;
    std::vector<std::string> files;
    int color;
};

void saveStackedComparisonHistogram(ROOT::RDF::RResultPtr<TH1D> realHist, std::vector<ROOT::RDF::RResultPtr<TH1D>>& mcHists, const std::vector<McSample>& mcSamples, const std::string& outputFile)
{
    TCanvas canvas("stackedComparisonCanvas", "stacked comparison histogram", 1200, 1000);

    TPad upperPad("upperPad", "histograms", 0.0, 0.30, 1.0, 1.0);
    TPad ratioPad("ratioPad", "ratio", 0.0, 0.0, 1.0, 0.30);

    upperPad.SetLeftMargin(0.14);
    upperPad.SetRightMargin(0.05);
    upperPad.SetTopMargin(0.08);
    upperPad.SetBottomMargin(0.02);

    ratioPad.SetLeftMargin(0.14);
    ratioPad.SetRightMargin(0.05);
    ratioPad.SetTopMargin(0.03);
    ratioPad.SetBottomMargin(0.35);

    upperPad.Draw();
    ratioPad.Draw();

    upperPad.cd();

    THStack mcStack("mcStack", realHist->GetTitle());
    TLegend legend(0.68, 0.60, 0.93, 0.90);
    legend.SetBorderSize(0);
    legend.SetFillStyle(0);

    TH1D totalMc(*mcHists.front().GetPtr());
    totalMc.SetDirectory(nullptr);
    totalMc.Reset();

    for (size_t i = 0; i < mcHists.size(); ++i) {
        TH1D* hist = mcHists[i].GetPtr();
        hist->SetFillColor(mcSamples[i].color);
        hist->SetLineColor(kBlack);
        hist->SetLineWidth(1);
        hist->SetStats(false);

        mcStack.Add(hist);
        totalMc.Add(hist);
        legend.AddEntry(hist, mcSamples[i].name.c_str(), "f");
    }

    realHist->SetStats(false);
    realHist->SetMarkerStyle(20);
    realHist->SetMarkerSize(0.8);
    realHist->SetLineColor(kBlack);

    const double maxY = std::max(realHist->GetMaximum(), totalMc.GetMaximum());
    mcStack.SetMaximum(maxY > 0.0 ? maxY * 1.35 : 1.0);
    mcStack.SetMinimum(0.0);

    mcStack.Draw("HIST");
    mcStack.GetXaxis()->SetLabelSize(0.0);
    mcStack.GetXaxis()->SetTitleSize(0.0);
    mcStack.GetYaxis()->SetTitle("Events");

    realHist->Draw("E1 SAME");
    legend.AddEntry(realHist.GetPtr(), "Real data", "lep");
    legend.Draw();

    ratioPad.cd();

    TH1D ratio(totalMc);
    ratio.SetDirectory(nullptr);
    ratio.SetName("mc_data_ratio");
    ratio.SetTitle("");
    ratio.SetStats(false);
    ratio.Divide(realHist.GetPtr());

    ratio.SetMinimum(0.0);
    ratio.SetMaximum(2.0);
    ratio.SetLineColor(kBlack);
    ratio.SetLineWidth(2);

    ratio.GetYaxis()->SetTitle("MC / Data");
    ratio.GetYaxis()->SetNdivisions(505);
    ratio.GetYaxis()->SetTitleSize(0.10);
    ratio.GetYaxis()->SetTitleOffset(0.55);
    ratio.GetYaxis()->SetLabelSize(0.085);

    ratio.GetXaxis()->SetTitle(realHist->GetXaxis()->GetTitle());
    ratio.GetXaxis()->SetTitleSize(0.12);
    ratio.GetXaxis()->SetTitleOffset(1.10);
    ratio.GetXaxis()->SetLabelSize(0.095);

    ratio.Draw("HIST");

    TLine unityLine(ratio.GetXaxis()->GetXmin(), 1.0, ratio.GetXaxis()->GetXmax(), 1.0);
    unityLine.SetLineColor(kGray + 2);
    unityLine.SetLineStyle(2);
    unityLine.Draw("SAME");

    canvas.Modified();
    canvas.Update();
    canvas.SaveAs(outputFile.c_str());

    logValue("Stacked comparison histogram saved", outputFile);
}

///Scatter matrix plot section
// void createFeatureMatrixMGL(
//     ROOT::RDF::RNode df,
//     std::vector<std::string> columns = {},
//     const std::string& output = "MlTemp/DY_training_feature_matrix.png",
//     int bins = 40
// ){
//     if(columns.empty()){
//         const std::string numeric =
//             " bool Bool_t char Char_t UChar_t short Short_t UShort_t "
//             "int Int_t UInt_t long Long_t ULong_t long long Long64_t "
//             "ULong64_t float Float_t double Double_t ";

//         for(const auto& c : df.GetColumnNames())
//             if(numeric.find(" " + df.GetColumnType(c) + " ") != std::string::npos)
//                 columns.push_back(c);
//     }

//     const int n = columns.size();
//     if(!n) return;

//     std::ostringstream e;
//     e << "ROOT::VecOps::RVec<double>{";
//     for(int i=0;i<n;i++) e << (i ? "," : "") << "double(" << columns[i] << ")";
//     e << "}";

//     auto result = df.Define("__mglMatrixRow",e.str())
//                     .Take<ROOT::VecOps::RVec<double>>("__mglMatrixRow");
//     const auto& rows = *result;

//     std::vector<std::vector<double>> v(n);
//     for(const auto& r : rows)
//         for(int i=0;i<n;i++)
//             if(std::isfinite(r[i])) v[i].push_back(r[i]);

//     std::vector<std::pair<double,double>> range(n);
//     for(int i=0;i<n;i++){
//         auto mm = std::minmax_element(v[i].begin(),v[i].end());
//         double a = v[i].empty() ? -1 : *mm.first;
//         double b = v[i].empty() ?  1 : *mm.second;
//         double p = a==b ? (a ? std::abs(a)*0.05 : 1) : (b-a)*0.05;
//         range[i] = {a-p,b+p};
//     }

//     mglGraph gr(std::max(800,240*n),std::max(800,240*n));
//     gr.Clf("w");

//     for(int y=0;y<n;y++) for(int x=0;x<=y;x++){
//         gr.SubPlot(n,n,y*n+x,"<_");
//         gr.SetRanges(range[x].first,range[x].second,
//                      x==y ? 0 : range[y].first,
//                      x==y ? 1 : range[y].second);

//         if(x==y){
//             std::vector<double> hx(bins), h(bins);
//             const double w=(range[x].second-range[x].first)/bins;

//             for(int k=0;k<bins;k++) hx[k]=range[x].first+(k+0.5)*w;
//             for(double z:v[x]){
//                 int k=(z-range[x].first)/w;
//                 if(k>=0 && k<bins) h[k]++;
//             }

//             const double top=h.empty()?1:*std::max_element(h.begin(),h.end());
//             gr.SetRanges(range[x].first,range[x].second,0,top*1.1+1);

//             mglData mx,my;
//             mx.Set(hx); my.Set(h);
//             gr.Axis("xy"); gr.Box(); gr.Bars(mx,my,"b");
//             gr.Label('x',columns[x].c_str(),0);
//         }else{
//             std::vector<double> px,py;
//             for(const auto& r:rows)
//                 if(std::isfinite(r[x]) && std::isfinite(r[y])){
//                     px.push_back(r[x]); py.push_back(r[y]);
//                 }

//             mglData mx,my;
//             mx.Set(px); my.Set(py);
//             gr.Axis("xy"); gr.Box();
//             gr.Plot(mx,my," k.");   // black dots, no connecting line
//             gr.Label('x',columns[x].c_str(),0);
//             gr.Label('y',columns[y].c_str(),0);
//         }
//     }

//     gr.WritePNG(output.c_str(),"",0,false);
// }

///// GRAPHS /////


//Uses the JSON for filter the luminosity blocks.
ROOT::RDF::RNode filterLuminosity(ROOT::RDF::RNode df, const std::string& fileName)
{
    std::ifstream f(fileName); // Reading the full JSON file
    std::string s((std::istreambuf_iterator<char>(f)), {});//This opens the JSON file and reads the whole file into one string called s.

    GoodLumiMapPtr good = std::make_shared<GoodLumiMap>();
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

//This function calculates the angular difference in φ between two particles. The angle φ is periodic. -π < φ ≤ π Foe example: φ1=3.13 φ2=-3.13.  Because they are close across the π/-π  then we use std::atan2(std::sin(a - b), std::cos(a - b))
float dphi(float a, float b) {
    return std::atan2(std::sin(a - b), std::cos(a - b));
}

//---This code implements the trigger-object matching part of the CMS Drell-Yan dimuon selection.---
//The event must pass the single-muon trigger. At least one selected reconstructed muon must match the HLT trigger object.
// In the event we can find several muons , and only some of them are good, but we do not remove data about bad muons. We anly filter the events. So in histogram we can find bad nuons as well
ROOT::RDF::RNode filterMuons(ROOT::RDF::RNode df){
    const std::string hltExpr = "HLT_IsoMu24"; //A isolated muon trigger (with a transverse-momentum threshold around 24 GeV) is an online selection system in CMS. It decides in real time whether the event is interesting enough to save. HLT_IsoMu20 OR HLT_IsoTkMu20 for the 2015 dataset.

    ROOT::RDF::RNode dfHlt = df.Filter(hltExpr);

    // Online means reconstruction done during data taking, while the LHC is producing collisions.
    // Offline means reconstruction done after the event has already been saved.
    auto dyDimuonFilter = [](const RVec<float>& muPt, //transverse momentum - Transverse momentum, particle’s momentum that is perpendicular to the proton beam direction.
                             const RVec<float>& muEta, //pseudorapidities - η, describes the direction of a particle relative to the proton beam axis. 0=perpendicular
                             const RVec<float>& muPhi, //azimuthal angles - φ, describes the direction of the particle around the beam axis in the transverse plane.
                             const RVec<int>& muCharge, //electric charges -1 or +1
                             const RVec<Bool_t>& muIsGlobal, // tells whether each muon is reconstructed as a global CMS muon using information from inner tracker and  outer muon chambers. (combines measurements from several detector systems.) 1 or 0
                             const RVec<float>& trigPt, //TrigObj_pt is the transverse momentum of an object reconstructed by the trigger system, not by the final offline reconstruction.
                             const RVec<float>& trigEta,
                             const RVec<float>& trigPhi,
                             const RVec<int>& trigId) {
        const size_t nMu = muPt.size();

        if (nMu < 2) return false; //Muon multiplicity: at least two muons

        for (size_t i = 0; i < nMu; ++i) {
            //Both must be global muons.
            //CMS cannot reconstruct muons equally well in all directions. The muon detector has reliable coverage approximately |η| < 2.4.
            if (!muIsGlobal[i]) continue;
            if (std::abs(muEta[i]) >= 2.4f) continue;

            for (size_t k = i + 1; k < nMu; ++k) {
                if (!muIsGlobal[k]) continue;
                if (std::abs(muEta[k]) >= 2.4f) continue;

                if (muCharge[i] * muCharge[k] >= 0) continue; //The intermediate particle is electrically neutral, so must be one positive muon and one negative muon.

                const float leadPt = muPt[i] > muPt[k] ? muPt[i] : muPt[k];
                const float subPt = muPt[i] > muPt[k] ? muPt[k] : muPt[i];

                if (leadPt <= 22.f) continue; // leading offline muon threshold
                if (subPt <= 10.f) continue; // subleading offline muon threshold

                for (size_t j = 0; j < trigPt.size(); ++j) {
                    if (std::abs(trigId[j]) != 13) continue; //The PDG ID of a muon is 13.
                    if (trigPt[j] <= 20.f) continue; // loose trigger-object pT requirement, not the offline leading-muon threshold.

                    const float dPhi1 = dphi(muPhi[i], trigPhi[j]);
                    const float dEta1 = muEta[i] - trigEta[j];
                    const float dR2_1 = dEta1 * dEta1 + dPhi1 * dPhi1;

                    const float dPhi2 = dphi(muPhi[k], trigPhi[j]);
                    const float dEta2 = muEta[k] - trigEta[j];
                    const float dR2_2 = dEta2 * dEta2 + dPhi2 * dPhi2;

                    if (dR2_1 < 0.01f || dR2_2 < 0.01f) return true; //ΔR is the angular distance between two objects in detector coordinates.
                }
            }
        }

        return false;
    };

    return dfHlt.Filter(
        dyDimuonFilter,
        {
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_charge",
            "Muon_isGlobal",
            "TrigObj_pt",
            "TrigObj_eta",
            "TrigObj_phi",
            "TrigObj_id"
        }
    );
}

//calculates the 3D angle between two reconstructed muon directions. eta — pseudorapidity phi — azimuthal angle The function calculates: α3D​=arccos(p1^​​⋅p2^​​)
float openingAngle3D(float eta1, float phi1, float eta2, float phi2){
    const float c = std::clamp(
        (std::cos(dphi(phi1, phi2)) + std::sinh(eta1) * std::sinh(eta2)) /
        (std::cosh(eta1) * std::cosh(eta2)),
        -1.0f,
        1.0f
    );

    return std::acos(c);
}

// A cosmic-ray muon is a muon coming from outside the collision, usually from cosmic-ray interactions in the atmosphere. It can pass through the CMS detector from top to bottom. The reconstruction can sometimes split this single cosmic muon into two tracks: one on one side of the detector and one on the opposite side. That fake pair looks like: μ1 ​and μ2​ and α3D​≈π


//These two function are implementing the muon-quality, isolation, and cosmic-ray rejection part of the CMS Drell–Yan dimuon selection.
// In the event we can find several muons , and only some of them are good, but we do not remove data about bad muons. We anly filter the events. So in histogram we can find bad nuons as well
ROOT::RDF::RNode filterMuonQualityAndCosmicRejection(ROOT::RDF::RNode df){
    auto pass = [](const RVec<float>& pt,
                   const RVec<float>& eta,
                   const RVec<float>& phi,
                   const RVec<int>& nStations,
                   const RVec<int>& nTrackerLayers,
                   const RVec<float>& dxy,
                   const RVec<float>& ptErr,
                   const RVec<float>& tkRelIso) {
        const size_t n = pt.size();

        for (size_t i = 0; i < n; ++i) {
            if (nStations[i] <= 1) continue; //A muon station is a layer/group of detector chambers in the CMS muon system. If a muon is seen in several stations, the reconstruction is more reliable.
            if (nTrackerLayers[i] <= 5) continue; //This requires the muon track to have hits in more than five tracker layers. The tracker is the inner CMS detector that measures charged-particle trajectories with high precision. More tracker layers mean the track is better constrained.
            if (std::abs(dxy[i]) >= 0.2f) continue; //dxy is the transverse impact parameter, meaning the closest distance between the muon track and the primary collision point in the transverse plane. The purpose is to select prompt muons, meaning muons that come from the proton-proton collision region, not from cosmic rays or displaced decays.
            if (pt[i] <= 0.f || ptErr[i] / pt[i] >= 0.3f) continue; //This is the relative transverse-momentum uncertainty. Drell–Yan invariant mass is reconstructed from the two muon momenta. So, if the muon momentum is badly measured, the dimuon mass can be badly distorted.
            if (tkRelIso[i] >= 0.1f) continue; //This is tracker isolation: muon is called isolated if there is not much other charged-particle activity around it.

            for (size_t j = i + 1; j < n; ++j) {
                if (nStations[j] <= 1) continue;
                if (nTrackerLayers[j] <= 5) continue;
                if (std::abs(dxy[j]) >= 0.2f) continue;
                if (pt[j] <= 0.f || ptErr[j] / pt[j] >= 0.3f) continue;
                if (tkRelIso[j] >= 0.1f) continue;

                if (std::abs(3.14159265358979323846f - openingAngle3D(eta[i], phi[i], eta[j], phi[j])) <= 0.005f) continue;//Cosmic rays cleaning ∣π−α3D​∣>0.005 0.005 rad=5 mrad

                return true;
            }
        }

        return false;
    };

    return df.Filter(
        pass,
        {
            "Muon_pt",
            "Muon_eta",
            "Muon_phi",
            "Muon_nStations",
            "Muon_nTrackerLayers",
            "Muon_dxy",
            "Muon_ptErr",
            "Muon_tkRelIso"
        }
    );
}

ROOT::RDF::RNode makeLeadingSubleadingMuonKinematicsDataFrame(ROOT::RDF::RNode df){
    return df
        .Define("LeadingSubleadingMuonKinematics",
            [](const RVec<float>& pt, const RVec<float>& eta, const RVec<float>& phi) {
                int lead = -1;
                int sub = -1;

                for (size_t i = 0; i < pt.size(); ++i) {
                    if (lead < 0 || pt[i] > pt[lead]) {
                        sub = lead;
                        lead = i;
                    } else if (sub < 0 || pt[i] > pt[sub]) {
                        sub = i;
                    }
                }

                if (lead < 0 || sub < 0) {
                    return RVec<float>{-1.f, 0.f, 0.f, -1.f, 0.f, 0.f};
                }

                return RVec<float>{
                    pt[lead], eta[lead], phi[lead],
                    pt[sub],  eta[sub],  phi[sub]
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
        .Filter("LeadingMuon_pt > 0.f && SubleadingMuon_pt > 0.f");
}

ROOT::RDF::RNode makeDimuonKinematicsDataFrame(ROOT::RDF::RNode df){
    return df
        .Define("DimuonKinematics",
            [](float leadPt, float leadEta, float leadPhi,
               float subPt, float subEta, float subPhi) {
                const ROOT::Math::PtEtaPhiMVector muLead(leadPt, leadEta, leadPhi, 0.105658f);//mμ​=0.105658 GeV is the muon mass.
                const ROOT::Math::PtEtaPhiMVector muSub(subPt, subEta, subPhi, 0.105658f); //mμ​=0.105658 GeV is the muon mass.
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

ROOT::RDF::RNode FullFilterDF(ROOT::RDF::RNode df, const std::string& eventWeightExpression) {
    return makeDimuonKinematicsDataFrame(
        makeLeadingSubleadingMuonKinematicsDataFrame(
            filterMuonQualityAndCosmicRejection(
                filterMuons(df)
            )
            .Define("eventWeight", eventWeightExpression)
            .Define("Muon_ptRelErr", "Muon_ptErr / Muon_pt")
        )
    );
}

/* EXPLANATION OF THE  FULLFILTERDF FUNCTION:
Initial event
{
  "event": 1001,
  "HLT_IsoMu24": true,

  "Muon_pt": [52.0, 45.0, 8.0],
  "Muon_eta": [0.40, -0.35, 1.10],
  "Muon_phi": [0.20, 3.10, -1.00],
  "Muon_charge": [1, -1, 1],
  "Muon_isGlobal": [true, true, false],

  "Muon_nStations": [2, 3, 1],
  "Muon_nTrackerLayers": [8, 7, 4],
  "Muon_dxy": [0.01, -0.02, 0.30],
  "Muon_ptErr": [1.0, 1.2, 3.0],
  "Muon_tkRelIso": [0.03, 0.05, 0.40],

  "TrigObj_pt": [50.0],
  "TrigObj_eta": [0.41],
  "TrigObj_phi": [0.21],
  "TrigObj_id": [13]
}

filterMuons(df)
HLT_IsoMu24 == true
at least 2 muons
both are global muons
both have ∣η∣<2.4
opposite charge q1​q2​<0
leading muon: pT​>22 GeV
subleading muon: pT​>10 GeV
trigger object must be a muon: abs(TrigObj_id) == 13
trigger  pT​>20 GeV
at least one candidate muon must match the trigger ΔR2<0.01

{
  "event": 1001,
  "...": "same columns as before"
}


filterMuonQualityAndCosmicRejection(df)  Function removes events that fail.
Muon_nStations > 1
Muon_nTrackerLayers > 5
|Muon_dxy| < 0.2
Muon_pt > 0
Muon_ptErr / Muon_pt < 0.3
Muon_tkRelIso < 0.1
And the pair must not look like a cosmic muon: ∣π−α3D​∣>0.005

{
  "event": 1001,
  "Muon_pt": [52.0, 45.0, 8.0],
  "...": "original columns remain"
}

Define("eventWeight", ...)   (For real data it can simply be: "eventWeight": 1.0)
{
  "event": 1001,
  "Muon_pt": [52.0, 45.0, 8.0],

  "eventWeight": 0.00342
}  

Define("Muon_ptRelErr", ...) 1.0,1.2,3.0]/[52,45,8] Muon_ptErr / Muon_pt
{
  "event": 1001,
  "Muon_pt": [52.0, 45.0, 8.0],
  "Muon_ptErr": [1.0, 1.2, 3.0],

  "eventWeight": 0.00342,
  "Muon_ptRelErr": [0.0192, 0.0267, 0.375] <= added column
}

makeLeadingSubleadingMuonKinematicsDataFrame()
Using "Muon_pt": [52.0, 45.0, 8.0] creates leading and subliading
{
  "LeadingMuon_pt": 52.0,
  "LeadingMuon_eta": 0.40,
  "LeadingMuon_phi": 0.20,

  "SubleadingMuon_pt": 45.0,
  "SubleadingMuon_eta": -0.35,
  "SubleadingMuon_phi": 3.10
}
Events are removed if: LeadingMuon_pt <= 0 or SubleadingMuon_pt <= 0

makeDimuonKinematicsDataFrame() From existing leading and subleading muons creates dimuon kinematics
{
  "Dimuon_mass": 96.3,
  "Dimuon_pt": 8.4,
  "Dimuon_rapidity": 0.06,
  "Dimuon_phi": 1.42
}
Finally .Filter("Dimuon_mass > 0.f")

Final df
{
  "event": 1001,

  "Muon_pt": [52.0, 45.0, 8.0],
  "Muon_eta": [0.40, -0.35, 1.10],
  "Muon_phi": [0.20, 3.10, -1.00],

  "eventWeight": 0.00342,

  "Muon_ptRelErr": [
    0.0192,
    0.0267,
    0.375
  ],

  "LeadingSubleadingMuonKinematics": [
    52.0,
    0.40,
    0.20,
    45.0,
    -0.35,
    3.10
  ],

  "LeadingMuon_pt": 52.0,
  "LeadingMuon_eta": 0.40,
  "LeadingMuon_phi": 0.20,

  "SubleadingMuon_pt": 45.0,
  "SubleadingMuon_eta": -0.35,
  "SubleadingMuon_phi": 3.10,

  "DimuonKinematics": [
    96.3,
    8.4,
    0.06,
    1.42
  ],

  "Dimuon_mass": 96.3,
  "Dimuon_pt": 8.4,
  "Dimuon_rapidity": 0.06,
  "Dimuon_phi": 1.42
}
*/

// docker start -ai cmscloud
// root -l -b -q -e '.L filter_dy1.C' -e 'singleAnalysis()'
void singleAnalysis() {
    resetExecutionTimer();
    resetLog();
 
    const bool isMonteCarlo = true;
    logValue("isMonteCarlo", isMonteCarlo); 
    const std::string RootDataFile = isMonteCarlo ? "dy_mc_main_xrootd_files.txt" : "files-recid-30563.txt";
    https://opendata.cern.ch/record/35671 and https://opendata.cern.ch/record/30563
    // "files-recid-30563.txt"; //real cms data
    // "dy_mc_main_xrootd_files.txt"; //mc simulation 

    const bool isJsonLuminosityExtraction = !isMonteCarlo;
    const std::string JsonLuminosityPath = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON_MuonPhys.txt";

    const std::string OutputPrefix = isMonteCarlo ? "mc" : "real";

    const bool DEV_MODE = true;
    const size_t DEV_N_FILES = 1;

    logValue("DEV_MODE", DEV_MODE); 
    if (DEV_MODE) {
        logValue("DEV_N_FILES", DEV_N_FILES); 
    }

    std::vector<std::string> allFiles; //files contains all ROOT file paths from your text file.  std::vector means a resizable array
    std::ifstream in(RootDataFile); //This opens the text file for reading.  std::ifstream means input file stream. It is used when you want to read data from a file.
    std::string path; //temporary string variable

    while (std::getline(in, path)) { //This reads the text file line by line
        if (!path.empty()) allFiles.push_back(path); //This checks whether the line is not empty if path is not empty - adds this path to the end of the vector.
    }
    std::vector<std::string> files = allFiles;
    logElapsedTime("Files loaded");

    if (DEV_MODE && files.size() > DEV_N_FILES) {
        files.resize(DEV_N_FILES);
    }

    logValue("Files used", files.size());

    ROOT::RDataFrame df("Events", files);
    //This line does not immediately loop over all events. RDataFrame builds a computation graph first. The actual event loop starts later when you request a result, for example with Count(), Snapshot(), or Histo1D(). ROOT calls such operations lazy actions, meaning they are booked first and executed only when their result is needed.

    // const size_t columnCount = df.GetColumnNames().size();
    // logValue("df column count", columnCount);
    logDataFrameShape("Initial df",df);
    // logMuonCountMinMax(df);

    logElapsedTime("filterLuminosity START");
    // ROOT::RDF::RNode dfGoodLumi = isJsonLuminosityExtraction ? filterLuminosity(df, JsonLuminosityPath) : df;
    ROOT::RDF::RNode dfGoodLumi = isJsonLuminosityExtraction ? filterLuminosity(df, JsonLuminosityPath) : df;
    // logDataFrameShape("dfGoodLumi", dfGoodLumi);
    logElapsedTime("filterLuminosity END");

    logElapsedTime("filterMuons START");
    ROOT::RDF::RNode dfGoodMuonsFirstLayer = filterMuons(dfGoodLumi);
    // logDataFrameShape("dfGoodMuonsFirstLayer",dfGoodMuonsFirstLayer);
    logElapsedTime("filterMuons END");

    logElapsedTime("filterMuonQualityAndCosmicRejection START");
    // ROOT::RDF::RNode DYdfNotNorm = filterMuonQualityAndCosmicRejection(dfGoodMuonsFirstLayer);
    ROOT::RDF::RNode DYdfNotNorm = filterMuonQualityAndCosmicRejection(dfGoodMuonsFirstLayer);
    // logDataFrameShape("DYdfNotNorm",DYdfNotNorm);
    logElapsedTime("filterMuonQualityAndCosmicRejection END");


       

    //I tested full scale data set and the filer was very small with JSON of good luminosity blocks.
    // std::cout << "All events: " << *df.Count() << "\n"; //All events: 174035164
    // std::cout << "Good lumi events: " << *dfGoodLumi.Count() << "\n"; //Good lumi events: 173467356

    // auto nProcessed = df.Count();
    // auto nGood = dfGoodLumi.Count();
    // auto nHltMatched = dfGoodMuonsFirstLayer.Count();

    // const ULong64_t vProcessed = *nProcessed; 
    // const ULong64_t vGood = *nGood; 
    // const ULong64_t vHltMatched = *nHltMatched; 

    // logLine("\n[Cut flow]"); 
    // logCutCount("Processed events", vProcessed); 
    // logCutCount("Good lumi events", vGood); 
    // logCutCount("HLT_IsoMu24 + matched muon events", vHltMatched); 

    /*
    run first unique values: 284038 283830 281976 282033 282034 ...
    luminosityBlock first unique values: 1 2 3 4 15 ...
    event first unique values: 1178292 1307941 1262974 366529 1436421 ...
    HLT_IsoMu24 first unique values: 0 1
    Muon_pt first unique values inside RVec: 23.1233 39.2836 24.118 91.9211 52.6816 ...
    Muon_eta first unique values inside RVec: 2.29883 1.79907 -1.7439 -1.22778 -1.23218 ...
    Muon_phi first unique values inside RVec: 0.417969 1.83179 -1.43896 0.957031 0.94104 ...
    Muon_charge first unique values inside RVec: 1 -1
    Muon_isGlobal first unique values inside RVec: 1 0
    TrigObj_pt first unique values inside RVec: 23.0898 23.3086 39.0859 24.0547 23.8789 ...
    TrigObj_eta first unique values inside RVec: 2.29785 1.7998 -1.74414 -1.22803 1.32568 ...
    TrigObj_phi first unique values inside RVec: 0.417969 0.417847 1.83154 -1.43848 0.956787 ...
    TrigObj_id first unique values inside RVec: 13 15 1 3 11
    */
    
    // logLine("\n[First unique values]");
    // logFirstUniqueValues<UInt_t>(df, "run"); 
    // logFirstUniqueValues<UInt_t>(df, "luminosityBlock"); 
    // logFirstUniqueValues<ULong64_t>(df, "event"); 
    // logFirstUniqueValues<Bool_t>(df, "HLT_IsoMu24"); 

    // logFirstUniqueValuesInRVec<float>(df, "Muon_pt");
    // logFirstUniqueValuesInRVec<float>(df, "Muon_eta");
    // logFirstUniqueValuesInRVec<float>(df, "Muon_phi");
    // logFirstUniqueValuesInRVec<int>(df, "Muon_charge");
    // logFirstUniqueValuesInRVec<Bool_t>(df, "Muon_isGlobal");

    // logFirstUniqueValuesInRVec<float>(df, "TrigObj_pt");
    // logFirstUniqueValuesInRVec<float>(df, "TrigObj_eta");
    // logFirstUniqueValuesInRVec<float>(df, "TrigObj_phi");
    // logFirstUniqueValuesInRVec<int>(df, "TrigObj_id");

    /*
    Muon_pt min: 3.00052 Muon_pt max: 81729.4
    Muon_eta min: -5.83984 Muon_eta max: 7.2959
    Muon_phi min: -3.1416 Muon_phi max: 3.1416
    Muon_charge min: -1 Muon_charge max: 1
    Muon_isGlobal min: 0 Muon_isGlobal max: 1
    TrigObj_pt min: 8.00195 TrigObj_pt max: 8318
    TrigObj_eta min: -3.91504 TrigObj_eta max: 4.08984
    TrigObj_phi min: -3.14062 TrigObj_phi max: 3.14062
    TrigObj_id min: 1 TrigObj_id max: 15
    */
    // logMinMaxValuesInRVec<float>(dfGoodMuonsFirstLayer, "Muon_pt");
    // logMinMaxValuesInRVec<float>(dfGoodMuonsFirstLayer, "Muon_eta");
    // logMinMaxValuesInRVec<float>(dfGoodMuonsFirstLayer, "Muon_phi");
    // logMinMaxValuesInRVec<int>(dfGoodMuonsFirstLayer, "Muon_charge");
    // logMinMaxValuesInRVec<Bool_t>(dfGoodMuonsFirstLayer, "Muon_isGlobal");

    // logMinMaxValuesInRVec<float>(dfGoodMuonsFirstLayer, "TrigObj_pt");
    // logMinMaxValuesInRVec<float>(dfGoodMuonsFirstLayer, "TrigObj_eta");
    // logMinMaxValuesInRVec<float>(dfGoodMuonsFirstLayer, "TrigObj_phi");
    // logMinMaxValuesInRVec<int>(dfGoodMuonsFirstLayer, "TrigObj_id");

    // logElapsedTime("Graphs START");
    // createScatterPlotFromRVec<float, float>(dfGoodMuonsFirstLayer, "Muon_eta", "Muon_pt", "muon_pt_vs_eta.png");
    // createScatterPlotFromRVec<float, float>(dfGoodMuonsFirstLayer, "Muon_phi", "Muon_pt", "muon_pt_vs_phi.png");
    // logElapsedTime("Graphs END");

    logElapsedTime("Normalization START");
    ROOT::RDF::RNode DYdfNorm = DYdfNotNorm.Define("eventWeight", "1.0");
    logElapsedTime("Normalization END");


    logElapsedTime("Df append and filter START");

    ROOT::RDF::RNode DYdfWithMuonRelErr = DYdfNorm.Define("Muon_ptRelErr", "Muon_ptErr / Muon_pt");

    ROOT::RDF::RNode DYdfWithLeadingSubleading = makeLeadingSubleadingMuonKinematicsDataFrame(DYdfWithMuonRelErr);

    ROOT::RDF::RNode DYdfPlot = makeDimuonKinematicsDataFrame(DYdfWithLeadingSubleading);

    logElapsedTime("Df append and filter END");

    // auto rawSelectedEvents = DYdfPlot.Count();
    // auto weightedSelectedEvents = DYdfPlot.Sum<double>("eventWeight");
    // auto weightedSelectedEvents2 = DYdfPlot
    //     .Define("eventWeight2", "eventWeight * eventWeight")
    //     .Sum<double>("eventWeight2");

    // logValue("Raw selected events after plot filters", *rawSelectedEvents);
    // logValue("Weighted selected events after plot filters", *weightedSelectedEvents);
    // logValue("Weighted statistical uncertainty after plot filters", std::sqrt(*weightedSelectedEvents2));
    


    logElapsedTime("Muon quality histograms START");

    auto hLeadingMuonPt = DYdfPlot.Histo1D({"hist_LeadingMuon_pt", "Leading muon pT;Leading muon pT [GeV];Events", 100, 0, 140}, "LeadingMuon_pt", "eventWeight");
    // auto hLeadingMuonEta = DYdfPlot.Histo1D({"hist_LeadingMuon_eta", "Leading muon eta;Leading muon eta;Events", 100, -2.4, 2.4}, "LeadingMuon_eta", "eventWeight");
    // auto hLeadingMuonPhi = DYdfPlot.Histo1D({"hist_LeadingMuon_phi", "Leading muon phi;Leading muon phi [rad];Events", 100, -3.2, 3.2}, "LeadingMuon_phi", "eventWeight");
    // logElapsedTime("Leading df");

    auto hSubleadingMuonPt = DYdfPlot.Histo1D({"hist_SubleadingMuon_pt", "Subleading muon pT;Subleading muon pT [GeV];Events", 100, 0, 140}, "SubleadingMuon_pt", "eventWeight");
    // auto hSubleadingMuonEta = DYdfPlot.Histo1D({"hist_SubleadingMuon_eta", "Subleading muon eta;Subleading muon eta;Events", 100, -2.4, 2.4}, "SubleadingMuon_eta", "eventWeight");
    // auto hSubleadingMuonPhi = DYdfPlot.Histo1D({"hist_SubleadingMuon_phi", "Subleading muon phi;Subleading muon phi [rad];Events", 100, -3.2, 3.2}, "SubleadingMuon_phi", "eventWeight");
    // logElapsedTime("Subleading df");

    // auto hDimuonMass = DYdfPlot.Histo1D({"hist_Dimuon_mass", "Dimuon invariant mass;Dimuon mass [GeV];Events", 100, 50, 150}, "Dimuon_mass", "eventWeight");
    // auto hDimuonPt = DYdfPlot.Histo1D({"hist_Dimuon_pt", "Dimuon pT;Dimuon pT [GeV];Events", 100, 0, 200}, "Dimuon_pt", "eventWeight");
    // auto hDimuonRapidity = DYdfPlot.Histo1D({"hist_Dimuon_rapidity", "Dimuon rapidity;Dimuon rapidity;Events", 100, -3, 3}, "Dimuon_rapidity", "eventWeight");
    // auto hDimuonPhi = DYdfPlot.Histo1D({"hist_Dimuon_phi", "Dimuon phi;Dimuon phi [rad];Events", 100, -3.2, 3.2}, "Dimuon_phi", "eventWeight");
    // logElapsedTime("Dimuon histograms");

    // saveHistogram(hLeadingMuonPt, "Images/" + OutputPrefix + "_hist_LeadingMuon_pt.png");// Units:  GeV
    // saveHistogram(hLeadingMuonEta, "Images/" + OutputPrefix + "_hist_LeadingMuon_eta.png");///eta, pseudorapidity, angular coordinate related to the polar angle relative to the beam axis. eta ≈ 0 => perpendicular to the beam.
    // saveHistogram(hLeadingMuonPhi, "Images/" + OutputPrefix + "_hist_LeadingMuon_phi.png");
 
    // logElapsedTime("Leading plot");
    saveHistogram(hSubleadingMuonPt, "Images/" + OutputPrefix + "_hist_SubleadingMuon_pt.png");
    // saveHistogram(hSubleadingMuonEta, "Images/" + OutputPrefix + "_hist_SubleadingMuon_eta.png");
    // saveHistogram(hSubleadingMuonPhi, "Images/" + OutputPrefix + "_hist_SubleadingMuon_phi.png");
    // logElapsedTime("Subleading plot");

    // saveHistogram(hDimuonMass, "Images/" + OutputPrefix + "_hist_Dimuon_mass.png");
    // saveHistogram(hDimuonPt, "Images/" + OutputPrefix + "_hist_Dimuon_pt.png");
    // saveHistogram(hDimuonRapidity, "Images/" + OutputPrefix + "_hist_Dimuon_rapidity.png");
    // saveHistogram(hDimuonPhi, "Images/" + OutputPrefix + "_hist_Dimuon_phi.png");
    // logElapsedTime("Dimuon plot");    

    createRVecHistogram<float>(DYdfPlot, "Muon_pt", 100, "Images/" + OutputPrefix + "_his_Muon_pt.png", 0, 140);
    // createRVecHistogram<float>(DYdfPlot, "Muon_eta", 100, "Images/" + OutputPrefix + "_his_Muon_eta.png");
    // createRVecHistogram<int>(DYdfPlot, "Muon_nStations", 100, "Images/" + OutputPrefix + "_his_Muon_nStations.png");
    // createRVecHistogram<int>(DYdfPlot, "Muon_nTrackerLayers", 100, "Images/" + OutputPrefix + "_his_Muon_nTrackerLayers.png");
    // createRVecHistogram<float>(DYdfPlot, "Muon_dxy", 100, "Images/" + OutputPrefix + "_his_Muon_dxy.png", -0.005, 0.005);
    // createRVecHistogram<float>(DYdfPlot, "Muon_ptRelErr", 100, "Images/" + OutputPrefix + "_his_Muon_ptRelErr.png", 0, 0.08);
    // createRVecHistogram<float>(DYdfPlot, "Muon_tkRelIso", 100, "Images/" + OutputPrefix + "_his_Muon_tkRelIso.png", 0.003, 0.07);

    
    //There are two different “directions” in the event: 1 The direction of each individual muon after the Z decay. 2 The direction of the whole dimuon system, which means the direction of the original Z/γ momentum. The produced Z boson can move in some direction because the two incoming partons usually do not carry equal momentum.  One quark may carry a larger proton momentum fraction x1 and the other may carry a smaller fraction x2.
    //Invariant mass is the total mass-energy of a system calculated from the energies and momenta of all its particles. It is called invariant because every observer, regardless of their velocity, calculates the same value. M^2=E^2−p^2. In my case the total energy and momentum are conserved. So: Mμμ​=MZ​.
    // Why the graphs is continuous: Unlike a real photon (which has zero mass), a virtual photon is an internal, off-shell particle that can temporarily carry almost any invariant mass allowed by the interaction. This is possible because it is not directly observed and exists only as an intermediate state. So the spectrum is continuous. 
    // Z boson lives only 3×10−25 s. Because of the uncertainty principle, ΔEΔt≥2/ℏ​, its energy (and therefore reconstructed mass) is not exactly fixed. This produces the familiar Breit–Wigner (Lorentzian) distribution around the nominal mass of about 91.2 GeV, with a natural width of approximately 2.5 GeV.
    //The transverse momentum of the entire dimuon system is: p_Tμμ​=∣p_​T1​+p_​T2​∣
    //Rapidity measures the motion of the whole dimuon system along the beam direction.

    logElapsedTime("Muon quality histograms END");

    logElapsedTime("Scatter plots START");
    // createScatterPlotFromRVec<float, float>(DYdfPlot, "Muon_eta", "Muon_pt", "Images/" + OutputPrefix + "_scatter_Muon_pt_vs_eta.png");
    // createScatterPlotFromRVec<float, float>(DYdfPlot, "Muon_eta", "Muon_phi", "Images/" + OutputPrefix + "_scatter_Muon_phi_vs_eta.png");
    // createScatterPlotFromRVec<float, float>(DYdfPlot, "Muon_tkRelIso", "Muon_dxy", "Images/" + OutputPrefix + "_scatter_Muon_dxy_vs_trackerIsolation.png");
    logElapsedTime("Scatter plots END");

    //The real data is the reference. You must normalize the MC so that it represents the same integrated luminosity and approximately the same pileup structure as the real dataset.
    // if (DEV_MODE && vHltMatched > 0) { 
    //     logFirstMatchedEvents(dfGoodMuonsFirstLayer, 5);
    // }

    logElapsedTime("Analysis finished");
}

/*
mc_his_Muon_dxy.png, real_his_Muon_dxy.png - Muon transverse impact parameter, Muon d_{xy} [cm]
mc_his_Muon_eta.png, real_his_Muon_eta.png - Muon pseudorapidity, Muon η
mc_his_Muon_nStations.png, real_his_Muon_nStations.png - Muon station multiplicity, Number of matched muon stations
mc_his_Muon_nTrackerLayers.png, real_his_Muon_nTrackerLayers.png - Muon tracker-layer multiplicity, Number of tracker layers
mc_his_Muon_pt.png, real_his_Muon_pt.png - Muon transverse momentum, Muon p_{T} [GeV]
mc_his_Muon_ptRelErr.png, real_his_Muon_ptRelErr.png - Muon relative transverse-momentum uncertainty, σ(p_{T}) / p_{T}
mc_his_Muon_tkRelIso.png, real_his_Muon_tkRelIso.png - Muon tracker relative isolation, Tracker relative isolation
mc_hist_Dimuon_mass.png, real_hist_Dimuon_mass.png - Dimuon invariant mass, M_{μμ} [GeV]
mc_hist_Dimuon_phi.png, real_hist_Dimuon_phi.png - Dimuon azimuthal angle, Dimuon φ [rad]
mc_hist_Dimuon_pt.png, real_hist_Dimuon_pt.png - Dimuon transverse momentum, Dimuon p_{T} [GeV]
mc_hist_Dimuon_rapidity.png, real_hist_Dimuon_rapidity.png - Dimuon rapidity, Dimuon rapidity
mc_hist_LeadingMuon_eta.png, real_hist_LeadingMuon_eta.png - Leading-muon pseudorapidity, Leading-muon η
mc_hist_LeadingMuon_phi.png, real_hist_LeadingMuon_phi.png - Leading-muon azimuthal angle, Leading-muon φ [rad]
mc_hist_LeadingMuon_pt.png, real_hist_LeadingMuon_pt.png - Leading-muon transverse momentum, Leading-muon p_{T} [GeV]
mc_hist_SubleadingMuon_eta.png, real_hist_SubleadingMuon_eta.png - Subleading-muon pseudorapidity, Subleading-muon η
mc_hist_SubleadingMuon_phi.png, real_hist_SubleadingMuon_phi.png - Subleading-muon azimuthal angle, Subleading-muon φ [rad]
mc_hist_SubleadingMuon_pt.png, real_hist_SubleadingMuon_pt.png - Subleading-muon transverse momentum, Subleading-muon p_{T} [GeV]
mc_scatter_Muon_dxy_vs_trackerIsolation.png, real_scatter_Muon_dxy_vs_trackerIsolation.png - Muon transverse impact parameter vs tracker isolation, Tracker relative isolation, Muon d_{xy} [cm]
mc_scatter_Muon_phi_vs_eta.png, real_scatter_Muon_phi_vs_eta.png - Muon azimuthal angle vs pseudorapidity, Muon η, Muon φ [rad]
mc_scatter_Muon_pt_vs_eta.png, real_scatter_Muon_pt_vs_eta.png - Muon transverse momentum vs pseudorapidity, Muon η, Muon p_{T} [GeV]
*/

////LUMINOSITY EVALUATION FOR THE REAL DATA AND NORMALIZATION
using RunLumiKey = std::pair<UInt_t, UInt_t>;
using LumiTable = std::map<RunLumiKey, double>;

//Just reading of the file
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

std::string makeMcWeight(double scale)
{
    std::ostringstream expression;
    expression << std::setprecision(17)
               << "static_cast<double>(genWeight) * " << scale;

    return expression.str();
}

std::vector<std::string> loadFileList(const std::string& fileListPath, bool devMode, size_t devNFiles){
    std::ifstream input(fileListPath);
    if (!input) throw std::runtime_error("Cannot open ROOT file list: " + fileListPath);

    std::vector<std::string> files;
    std::string path;

    while (std::getline(input, path)) {
        while (!path.empty() && std::isspace(static_cast<unsigned char>(path.back()))) path.pop_back();
        if (path.empty()) continue;

        std::unique_ptr<TFile> testFile(TFile::Open(path.c_str(), "READ"));//Sometime s the cern storage have missing files and then we have to skip them.
        if (!testFile || testFile->IsZombie()) { logLine("Skipping inaccessible ROOT file: " + path); continue; }

        files.push_back(path);
        if (devMode && files.size() >= devNFiles) break;
    }

    if (files.empty()) throw std::runtime_error("No accessible ROOT files found in: " + fileListPath);
    return files;
}

struct BackgroundInput {
    std::string name;
    std::string fileList;
    double crossSectionPb;
};

struct PlotSpec {
    std::string name;
    std::string column;
    std::string title;
    std::string xTitle;
    int bins;
    double xMin;
    double xMax;
};

struct ComparisonPlot {
    PlotSpec spec;
    ROOT::RDF::RResultPtr<TH1D> realHist;
    std::vector<ROOT::RDF::RResultPtr<TH1D>> mcHists;
};
/////

// docker start -ai cmscloud
// root -l -b -q -e '.L filter_dy1.C' -e 'comparisonAnalysis()'
void similar as in the comparisonAnalysis()
{
    resetExecutionTimer();
    resetLog();
    ROOT::EnableImplicitMT(4);// parralel event processing for hystograms.

    const std::string RealRootDataFile = "files-recid-30563.txt";
    const std::string McMainRootDataFile = "dy_mc_main_xrootd_files.txt";
    const std::string JsonLuminosityPath = "Cert_271036-284044_13TeV_Legacy2016_Collisions16_JSON_MuonPhys.txt";
    const std::string LuminosityCsvPath = "Run2016H_MuonPhys_lumibyls.csv";
    
    const bool DEV_MODE = true;
    const size_t DEV_N_FILES = 2;
    
    logValue("comparisonAnalysis", true);
    logValue("DEV_MODE", DEV_MODE);
    if (DEV_MODE) {logValue("DEV_N_FILES", DEV_N_FILES);}

    const double McMainCrossSectionPb = 5396.0; // from related dataset https://opendata.cern.ch/record/35670

    const std::string BackgroundListDir = "Background_dataset_files/";

    const std::vector<BackgroundInput> backgroundInputs = {
        {"ttbar", BackgroundListDir + "ttbar_xrootd_files.txt", 687.1},// https://opendata.cern.ch/record/67801
        {"single_top_tw", BackgroundListDir + "single_top_tw_xrootd_files.txt", 32.45},// https://opendata.cern.ch/record/64881
        {"single_antitop_tw", BackgroundListDir + "single_antitop_tw_xrootd_files.txt", 32.51},// https://opendata.cern.ch/record/64825
        {"ww", BackgroundListDir + "ww_xrootd_files.txt", 75.88},// https://opendata.cern.ch/record/72696
        {"wz", BackgroundListDir + "wz_xrootd_files.txt", 27.56},// https://opendata.cern.ch/record/72754
        {"zz", BackgroundListDir + "zz_xrootd_files.txt", 12.08},// https://opendata.cern.ch/record/75593
        {"wjets", BackgroundListDir + "wjets_xrootd_files.txt", 54110.0},// https://opendata.cern.ch/record/69747
        {"qcd_pt20to30", BackgroundListDir + "qcd_pt20to30_xrootd_files.txt", 2.529e6}// https://opendata.cern.ch/record/cms-63186
        // {"photon_induced_dimuon", BackgroundListDir + "photon_induced_dimuon_xrootd_files.txt", ????}
    };

    const std::vector<std::string> realFiles = loadFileList(RealRootDataFile, DEV_MODE, DEV_N_FILES);
    const std::vector<std::string> mcMainFiles = loadFileList(McMainRootDataFile, DEV_MODE, DEV_N_FILES);
    const double IntegratedLuminosityFb = calculateIntegratedLuminosityFb(realFiles, JsonLuminosityPath, LuminosityCsvPath);

    ROOT::RDataFrame realDf("Events", realFiles);// creates a ROOT RDataFrame "realDf" that reads the Events tree from the ROOT files listed in realFiles
    logDataFrameShape("Final realDf", realDf);//Main Filter of the real data.
    ROOT::RDF::RNode realFilteredDf = FullFilterDF(filterLuminosity(realDf, JsonLuminosityPath), "1.0");
    logDataFrameShape("Final realFilteredDf", realFilteredDf);

    std::vector<McSample> mcSamples;
    std::vector<std::string> allMcFiles;

    const auto addMcSample = [&](int id, const std::string& name, const std::vector<std::string>& files, double crossSectionPb, int color) {
        const double normalizationScale = calculateNormalization(name, files, crossSectionPb, IntegratedLuminosityFb);
        mcSamples.push_back({id, name, crossSectionPb, normalizationScale, files, color});
        allMcFiles.insert(allMcFiles.end(), files.begin(), files.end());
    };

    addMcSample(0, "main_dy", mcMainFiles, McMainCrossSectionPb, kAzure - 9);

    int nextSampleId = 1;
    const std::vector<int> backgroundColors = {kOrange - 3, kGreen + 2, kSpring - 6, kViolet - 6, kMagenta - 7, kRed - 7, kCyan - 6, kGray + 1};

    for (size_t i = 0; i < backgroundInputs.size(); ++i) {
        const auto& input = backgroundInputs[i];
        const std::vector<std::string> files =loadFileList(input.fileList, DEV_MODE, DEV_N_FILES);

        addMcSample(nextSampleId++, input.name, files, input.crossSectionPb, backgroundColors[i % backgroundColors.size()]);
    }

    const auto findSample = [mcSamples](const ROOT::RDF::RSampleInfo& sampleInfo) {
        for (const auto& sample : mcSamples) {
            for (const auto& file : sample.files) {
                if (sampleInfo.Contains(file)) return sample;
            }
        }

        throw std::runtime_error("No MC sample found for file: " + sampleInfo.AsString());
    };

    ROOT::RDataFrame mcCombinedDf("Events", allMcFiles);

    auto mcCombinedDfWithSampleInfo = mcCombinedDf
        .DefinePerSample("mcSampleId", [findSample](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
            return findSample(sampleInfo).id;
        })
        .DefinePerSample("mcSampleName", [findSample](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
            return findSample(sampleInfo).name;
        })
        .DefinePerSample("sampleNormalization", [findSample](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
            return findSample(sampleInfo).normalizationScale;
        });

    ROOT::RDF::RNode mcCombinedFilteredDf = FullFilterDF(mcCombinedDfWithSampleInfo, "static_cast<double>(genWeight) * sampleNormalization");

    // const double mcMainNormalizationScale = calculateNormalization("main_dy", mcMainFiles, McMainCrossSectionPb, IntegratedLuminosityFb);
    // mcSamples.push_back({"main_dy", McMainCrossSectionPb, mcMainNormalizationScale, mcMainFiles});
    // allMcFiles.insert(allMcFiles.end(), mcMainFiles.begin(), mcMainFiles.end());

    // for (const auto& input : backgroundInputs) {
    //     const std::vector<std::string> files = loadFileList(input.fileList, DEV_MODE, DEV_N_FILES);
    //     const double normalizationScale = calculateNormalization(input.name, files, input.crossSectionPb, IntegratedLuminosityFb);
    //     mcSamples.push_back({input.name, input.crossSectionPb, normalizationScale, files});
    //     allMcFiles.insert(allMcFiles.end(), files.begin(), files.end());
    // }

    // ROOT::RDataFrame mcCombinedDf("Events", allMcFiles);

    // auto mcCombinedDfWithNormalization = mcCombinedDf.DefinePerSample("sampleNormalization", [mcSamples](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
    //     for (const auto& sample : mcSamples) {
    //         for (const auto& file : sample.files) {
    //             if (sampleInfo.Contains(file)) return sample.normalizationScale;
    //         }
    //     }
    //     throw std::runtime_error("No normalization found for MC file: " + sampleInfo.AsString());
    // });

    logElapsedTime("Real and combined MC dataframes ready");



    logElapsedTime("Comparison histograms START");

    const std::vector<PlotSpec> plotSpecs = {
        // {"LeadingMuon_pt", "LeadingMuon_pt", "Leading muon pT", "Leading muon pT [GeV]", 100, 0, 140},
        // {"LeadingMuon_eta", "LeadingMuon_eta", "Leading muon eta", "Leading muon eta", 100, -2.4, 2.4},
        // {"LeadingMuon_phi", "LeadingMuon_phi", "Leading muon phi", "Leading muon phi [rad]", 100, -3.2, 3.2},
        // {"SubleadingMuon_pt", "SubleadingMuon_pt", "Subleading muon pT", "Subleading muon pT [GeV]", 100, 0, 140},
        // {"SubleadingMuon_eta", "SubleadingMuon_eta", "Subleading muon eta", "Subleading muon eta", 100, -2.4, 2.4},
        // {"SubleadingMuon_phi", "SubleadingMuon_phi", "Subleading muon phi", "Subleading muon phi [rad]", 100, -3.2, 3.2},
        {"Dimuon_mass", "Dimuon_mass", "Dimuon invariant mass", "Dimuon mass [GeV]", 100, 50, 150}
        // {"Dimuon_pt", "Dimuon_pt", "Dimuon pT", "Dimuon pT [GeV]", 100, 0, 200},
        // {"Dimuon_rapidity", "Dimuon_rapidity", "Dimuon rapidity", "Dimuon rapidity", 100, -3, 3},
        // {"Dimuon_phi", "Dimuon_phi", "Dimuon phi", "Dimuon phi [rad]", 100, -3.2, 3.2},
        // {"Muon_pt", "Muon_pt", "Muon pT", "Muon pT [GeV]", 100, 0, 140}
    };

    
    std::vector<ComparisonPlot> plots;
    plots.reserve(plotSpecs.size());

    for (const auto& spec : plotSpecs) {
        const std::string title = spec.title + ";" + spec.xTitle + ";Events";
        const std::string realName = "real_" + spec.name;

        auto realHist =
            realFilteredDf.Histo1D({realName.c_str(), title.c_str(), spec.bins, spec.xMin, spec.xMax},
                                spec.column,
                                "eventWeight");

        std::vector<ROOT::RDF::RResultPtr<TH1D>> mcHists;

        for (const auto& sample : mcSamples) {
            const std::string mcName = "mc_" + sample.name + "_" + spec.name;

            auto sampleDf =
                mcCombinedFilteredDf.Filter([sampleId = sample.id](int mcSampleId) {
                    return mcSampleId == sampleId;
                }, {"mcSampleId"});

            auto mcHist =
                sampleDf.Histo1D({mcName.c_str(), title.c_str(), spec.bins, spec.xMin, spec.xMax},
                                spec.column,
                                "eventWeight");

            mcHists.push_back(mcHist);
        }

        plots.push_back({spec, realHist, mcHists});
    }

    std::vector<ROOT::RDF::RResultHandle> histogramHandles;

    for (auto& plot : plots) {
        histogramHandles.emplace_back(plot.realHist);

        for (auto& mcHist : plot.mcHists) {
            histogramHandles.emplace_back(mcHist);
        }
    }

    ROOT::RDF::RunGraphs(histogramHandles);

    for (auto& plot : plots) {
        saveStackedComparisonHistogram(
            plot.realHist,
            plot.mcHists,
            mcSamples,
            "Images/comparison_" + plot.spec.name + ".png"
        );
    }

    logElapsedTime("Comparison histograms END");
}


ROOT::RDF::RNode makeMlRawMuonFeaturesDataFrame(ROOT::RDF::RNode df, size_t maxMuons)
{
    ROOT::RDF::RNode out = df
        .Filter([maxMuons](const RVec<float>& pt) { return pt.size() <= maxMuons; }, {"Muon_pt"})
        .Define("MuonCount", [](const RVec<float>& pt) { return static_cast<float>(pt.size()); }, {"Muon_pt"})
        .Define("HLT_IsoMu24_f", "static_cast<float>(HLT_IsoMu24)");

    const std::vector<std::pair<std::string, std::string>> fields = {
        {"pt", "Muon_pt"}, {"eta", "Muon_eta"}, {"phi", "Muon_phi"}, {"charge", "Muon_charge"},
        {"isGlobal", "Muon_isGlobal"}, {"nStations", "Muon_nStations"}, {"nTrackerLayers", "Muon_nTrackerLayers"},
        {"dxy", "Muon_dxy"}, {"ptErr", "Muon_ptErr"}, {"tkRelIso", "Muon_tkRelIso"}
    };

    for (size_t i = 0; i < maxMuons; ++i) {
        const std::string index = std::to_string(i);
        for (const auto& [name, branch] : fields)
            out = out.Define("Muon" + index + "_" + name, branch + ".size() > " + index + " ? static_cast<float>(" + branch + "[" + index + "]) : -999.f");
    }

    return out;
}

// ROOT::RDF::RNode makeMlMuonQualityFeaturesDataFrame(ROOT::RDF::RNode df)
// {
//     return df
//         .Define("MlLeadingMuonIndex", [](const RVec<float>& pt) {
//             if (pt.size() < 2) return -1;
//             int lead = 0;
//             for (size_t i = 1; i < pt.size(); ++i)
//                 if (pt[i] > pt[lead]) lead = static_cast<int>(i);
//             return lead;
//         }, {"Muon_pt"})

//         .Define("MlSubleadingMuonIndex", [](const RVec<float>& pt, int lead) {
//             if (pt.size() < 2 || lead < 0) return -1;

//             int sub = -1;
//             for (size_t i = 0; i < pt.size(); ++i) {
//                 if (static_cast<int>(i) == lead) continue;
//                 if (sub < 0 || pt[i] > pt[sub]) sub = static_cast<int>(i);
//             }
//             return sub;
//         }, {"Muon_pt", "MlLeadingMuonIndex"})

//         .Filter("MlLeadingMuonIndex >= 0 && MlSubleadingMuonIndex >= 0")

//         .Define("LeadingMuon_tkRelIso", "Muon_tkRelIso[MlLeadingMuonIndex]")
//         .Define("SubleadingMuon_tkRelIso", "Muon_tkRelIso[MlSubleadingMuonIndex]")

//         .Define("LeadingMuon_dxy", "Muon_dxy[MlLeadingMuonIndex]")
//         .Define("SubleadingMuon_dxy", "Muon_dxy[MlSubleadingMuonIndex]")

//         .Define("LeadingMuon_ptRelErr",
//                 "Muon_pt[MlLeadingMuonIndex] > 0.f ? Muon_ptErr[MlLeadingMuonIndex] / Muon_pt[MlLeadingMuonIndex] : 999.f")
//         .Define("SubleadingMuon_ptRelErr",
//                 "Muon_pt[MlSubleadingMuonIndex] > 0.f ? Muon_ptErr[MlSubleadingMuonIndex] / Muon_pt[MlSubleadingMuonIndex] : 999.f")

//         .Define("LeadingMuon_nStations",
//                 "static_cast<float>(Muon_nStations[MlLeadingMuonIndex])")
//         .Define("SubleadingMuon_nStations",
//                 "static_cast<float>(Muon_nStations[MlSubleadingMuonIndex])")

//         .Define("LeadingMuon_nTrackerLayers",
//                 "static_cast<float>(Muon_nTrackerLayers[MlLeadingMuonIndex])")
//         .Define("SubleadingMuon_nTrackerLayers",
//                 "static_cast<float>(Muon_nTrackerLayers[MlSubleadingMuonIndex])");
// }

// ROOT::RDF::RNode makeMlDimuonAngularFeaturesDataFrame(ROOT::RDF::RNode df)
// {
//     return df
//         .Define("Dimuon_deltaPhi",
//             [](float phi1, float phi2) {
//                 return std::abs(dphi(phi1, phi2));
//             },
//             {"LeadingMuon_phi", "SubleadingMuon_phi"})

//         .Define("Dimuon_deltaR",
//             [](float eta1, float phi1, float eta2, float phi2) {
//                 const float dEta = eta1 - eta2;
//                 const float dPhi = dphi(phi1, phi2);
//                 return std::sqrt(dEta * dEta + dPhi * dPhi);
//             },
//             {"LeadingMuon_eta", "LeadingMuon_phi",
//              "SubleadingMuon_eta", "SubleadingMuon_phi"})

//         .Define("Dimuon_openingAngle3D",
//             [](float eta1, float phi1, float eta2, float phi2) {
//                 return openingAngle3D(eta1, phi1, eta2, phi2);
//             },
//             {"LeadingMuon_eta", "LeadingMuon_phi",
//              "SubleadingMuon_eta", "SubleadingMuon_phi"});
// }



// docker desktop start
// docker stats cmscloud
//docker stop cmscloud OR //exit
// docker start -ai cmscloud
// root -l -b -q -e '.L filter_dy1.C' -e 'MlRun()'
void MlRun()
{
    std::cout << "MlRun started2\n" << std::flush;
    resetExecutionTimer();
    resetLog();
    gSystem->mkdir("MlTemp", true);

    const unsigned int numberOfThreads = 4;
    const bool DevelopmentMode = true;
    const size_t DevelopmentFilesPerSample = 1;
    const bool UsePreparedData = false; // false = create training/testing ROOT files, true = load them
    const double trainingFraction = 0.70;

    const std::string TrainingDatasetPath = "MlTemp/ML_DY_training.root";
    const std::string TestingDatasetPath = "MlTemp/ML_DY_testing.root";
    // const std::string DyFileListPath = "dy_mc_main_xrootd_files.txt";
    const std::vector<std::string> BackgroundFileListPaths = {
        "dy_mc_main_xrootd_files.txt",
        "Background_dataset_files/ttbar_xrootd_files.txt",
        "Background_dataset_files/single_top_tw_xrootd_files.txt",
        "Background_dataset_files/single_antitop_tw_xrootd_files.txt",
        "Background_dataset_files/ww_xrootd_files.txt",
        "Background_dataset_files/wz_xrootd_files.txt",
        "Background_dataset_files/zz_xrootd_files.txt",
        "Background_dataset_files/wjets_xrootd_files.txt",
        "Background_dataset_files/qcd_pt20to30_xrootd_files.txt"
    };

    const size_t MaxMuons = 2;//The dat is cutted up to N muons.

    // const std::vector<std::string> features = [MaxMuons]() {
    //     std::vector<std::string> f = {
    //         "MuonCount",
    //         "HLT_IsoMu24_f",
    //         "MET_pt",
    //         "HasMuonTrigger",
    //         "MinMuonTriggerDeltaR2",
    //         "MaxMuonTriggerPt"
    //     };

    //     const std::vector<std::string> fields = {
    //         "pt",
    //         "eta",
    //         "charge",
    //         "isGlobal",
    //         "nStations",
    //         "nTrackerLayers",
    //         "dxy",
    //         "ptErr",
    //         "tkRelIso"
    //     };

    //     for (size_t i = 0; i < MaxMuons; ++i)
    //         for (const std::string& field : fields)
    //             f.push_back("Muon" + std::to_string(i) + "_" + field);

    //     return f;
    // }();


    const unsigned int numberOfTrees = 60;
    const unsigned int maximumDepth = 6;//A depth-three tree can represent interactions among several variables.
    const double minimumNodeSizePercent = 0.005;//This controls how small a tree node is allowed to become.
    const double baggedSampleFraction = 0.3;//It can reduce correlations between trees and decrease overfitting. PVZ Only 20 % is used
    const unsigned int numberOfCuts = 30;//More cuts can improve split resolution but increase computation. TMVA evaluates a finite set of candidate positions
    const double shrinkage = 0.05;//It is the gradient boosting learning rate
    const unsigned int randomSeed = 1;
    const double decisionThreshold = 0.0;//decisionThreshold is the cutoff applied to the BDT output score to convert the continuous BDT prediction into a binary class.

    logLine("--- ML RUN SETTINGS ---");
    logValue("Use prepared data", UsePreparedData);
    logValue("Training dataset", TrainingDatasetPath);
    logValue("Testing dataset", TestingDatasetPath);
    logValue("Training fraction", trainingFraction);
    logValue("Testing fraction", 1.0 - trainingFraction);
    logValue("Development mode", DevelopmentMode);
    if (DevelopmentMode) logValue("Development files per sample", DevelopmentFilesPerSample);
    logValue("Training threads", numberOfThreads);
    // logLine("[Features]");
    // for (const std::string& feature : features) logValue("Feature", feature);
    logLine("[BDT model parameters]");
    logValue("Method", "BDTG"); logValue("Boost type", "Gradient"); logValue("Number of trees", numberOfTrees);
    logValue("Maximum depth", maximumDepth); logValue("Minimum node size [%]", minimumNodeSizePercent);
    logValue("Bagged sample fraction", baggedSampleFraction); logValue("Number of cuts", numberOfCuts);
    logValue("Shrinkage", shrinkage); logValue("Random seed", randomSeed); logValue("Decision threshold", decisionThreshold);
    logLine("------");

    if (!UsePreparedData) {
        logLine("--- CREATE TRAINING AND TESTING DATA ---");

        // const std::vector<std::string> dyFiles = loadFileList(DyFileListPath, DevelopmentMode, DevelopmentFilesPerSample);
        std::vector<std::string> allFiles;
        for (const std::string& fileList : BackgroundFileListPaths) {
            const auto files = loadFileList(fileList, DevelopmentMode, DevelopmentFilesPerSample);
            allFiles.insert(allFiles.end(), files.begin(), files.end());
        }
        std::cout << "allFiles size: " << allFiles.size() << '\n';

        for (size_t i = 0; i < allFiles.size(); ++i)
            std::cout << "[" << i << "] " << allFiles[i] << '\n';

        // logValue("DY files", dyFiles.size());
        // logValue("All MC files", allFiles.size());

//         ROOT::RDataFrame dyDf("Events", dyFiles);
//         const ULong64_t numberOfDyEvents = *dyDf.Count();

//         auto passedResult = FullFilterDF(
//             dyDf.Define("mlEntry", "rdfentry_").Filter([MaxMuons](const RVec<float>& pt) { return pt.size() <= MaxMuons; }, {"Muon_pt"}),
//             "1.0"
//         ).Take<ULong64_t>("mlEntry");

//         const auto passedDyEntries = std::make_shared<std::unordered_set<ULong64_t>>(passedResult->begin(), passedResult->end());

//         logValue("DY events", numberOfDyEvents);
//         logValue("DY events passing FullFilterDF", passedDyEntries->size());

//         ROOT::RDataFrame rawMcDf("Events", allFiles);

//         auto labeledDf = makeMlRawMuonFeaturesDataFrame(rawMcDf.Define("mlEntry", "rdfentry_"), MaxMuons)
//             .Define("IsDyProcess", [passedDyEntries, numberOfDyEvents](ULong64_t entry) {
//                 return entry < numberOfDyEvents && passedDyEntries->find(entry) != passedDyEntries->end() ? 1 : 0;
//             }, {"mlEntry"});

//         const ULong64_t splitModulo = 10000;
//         const ULong64_t trainLimit = static_cast<ULong64_t>(trainingFraction * splitModulo);

//         auto trainingNode = labeledDf.Filter([trainLimit](ULong64_t entry) { return entry % 10000 < trainLimit; }, {"mlEntry"});
//         auto testingNode = labeledDf.Filter([trainLimit](ULong64_t entry) { return entry % 10000 >= trainLimit; }, {"mlEntry"});

//         std::vector<std::string> outputColumns = features;
//         outputColumns.insert(outputColumns.end(), {"event", "IsDyProcess"});

//         std::cout << "Separating muon columns and writing training/testing datasets...\n";
//         ROOT::RDF::Experimental::AddProgressBar(labeledDf);
//         ROOT::RDF::RSnapshotOptions snapshotOptions;
//         snapshotOptions.fLazy = true;

//         auto trainingSnapshot = trainingNode.Snapshot(
//             "Events",
//             TrainingDatasetPath,
//             outputColumns,
//             snapshotOptions
//         );

//         auto testingSnapshot = testingNode.Snapshot(
//             "Events",
//             TestingDatasetPath,
//             outputColumns,
//             snapshotOptions
//         );

//         ROOT::RDF::RunGraphs({trainingSnapshot, testingSnapshot});


//         logElapsedTime("Training/testing files created");
    } else {
        // logLine("--- LOAD EXISTING TRAINING AND TESTING DATA ---");
        // logElapsedTime("Prepared datasets selected");
    }

//     ROOT::RDataFrame trainingDf("Events", TrainingDatasetPath);
//     ROOT::RDataFrame testingDf("Events", TestingDatasetPath);
//     auto trainingSignal = trainingDf.Filter("IsDyProcess == 1").Count();
//     auto trainingBackground = trainingDf.Filter("IsDyProcess == 0").Count();
//     auto testingSignal = testingDf.Filter("IsDyProcess == 1").Count();
//     auto testingBackground = testingDf.Filter("IsDyProcess == 0").Count();
//     std::vector<ROOT::RDF::RResultHandle> countHandles;
//     countHandles.emplace_back(trainingSignal); countHandles.emplace_back(trainingBackground); countHandles.emplace_back(testingSignal); countHandles.emplace_back(testingBackground);
//     ROOT::RDF::RunGraphs(countHandles);
//     logValue("Training signal", *trainingSignal); logValue("Training background", *trainingBackground);
//     logValue("Testing signal", *testingSignal); logValue("Testing background", *testingBackground);

//     const std::string weightsFile = trainDrellYanBdt(features, "Events", TrainingDatasetPath, TestingDatasetPath,
//         "MlTemp/TMVA_DY_training.root", "MlTemp/BDT_training_history.root", "DrellYanClassification", "BDTG", "DY_BDT_dataset",
//         numberOfTrees, maximumDepth, minimumNodeSizePercent, baggedSampleFraction, numberOfCuts, shrinkage, randomSeed);

//     logValue("Trained model", weightsFile); logElapsedTime("BDT training END");

//     evaluateDrellYanBdt(features, "Events", TestingDatasetPath, weightsFile, "BDTG", "MlTemp/ML_DY_evaluation.root", "Predictions", decisionThreshold, LOG_FILE);
//     logElapsedTime("BDT evaluation END");
//     logElapsedTime("MlRun complete");
}









// void MlRun()
// {
//     resetExecutionTimer();
//     resetLog();
//     gSystem->mkdir("MlTemp", true);

//     const unsigned int numberOfThreads = 4;
//     const bool DevelopmentMode = true;
//     const size_t DevelopmentFilesPerSample = 3;
//     const bool UsePreparedData = false; // first run false; after correct files are created -> true
//     const double trainingFraction = 0.70;

//     const size_t MaxMuons = 4;
//     const size_t ModelMuons = 2;

//     const std::string TrainingDatasetPath = "MlTemp/ML_DY_training.root";
//     const std::string TestingDatasetPath = "MlTemp/ML_DY_testing.root";
//     const std::string DyFileListPath = "dy_mc_main_xrootd_files.txt";

//     const std::vector<std::string> BackgroundFileListPaths = {
//         "Background_dataset_files/ttbar_xrootd_files.txt",
//         "Background_dataset_files/single_top_tw_xrootd_files.txt",
//         "Background_dataset_files/single_antitop_tw_xrootd_files.txt",
//         "Background_dataset_files/ww_xrootd_files.txt",
//         "Background_dataset_files/wz_xrootd_files.txt",
//         "Background_dataset_files/zz_xrootd_files.txt",
//         "Background_dataset_files/wjets_xrootd_files.txt",
//         "Background_dataset_files/qcd_pt20to30_xrootd_files.txt"
//     };

//     const std::vector<std::string> muonFields = {
//         "pt", "eta", "phi", "charge", "isGlobal",
//         "nStations", "nTrackerLayers", "dxy", "ptRelErr", "tkRelIso"
//     };

//     const std::vector<std::string> features = [ModelMuons, muonFields]() {//is a C++ lambda function that is immediately executed. [captured_variables](parameters) {body }(arguments);
//         std::vector<std::string> f = {"MuonCount", "MET_pt"};
//         for (size_t i = 0; i < ModelMuons; ++i)
//             for (const auto& field : muonFields)
//                 f.push_back("Muon" + std::to_string(i) + "_" + field);

//         f.insert(f.end(), {
//             "Dimuon_mass",
//             "Dimuon_deltaPhi",
//             "Dimuon_deltaR",
//             "Dimuon_openingAngle3D"
//         });

//         return f;
//     }();

//     const unsigned int numberOfTrees = 100;
//     const unsigned int maximumDepth = 4;
//     const double minimumNodeSizePercent = 0.01;
//     const double baggedSampleFraction = 0.5;
//     const unsigned int numberOfCuts = 50;
//     const double shrinkage = 0.05;
//     const unsigned int randomSeed = 1;
//     const double decisionThreshold = 0.0; // baseline only; tune later for high precision

//     ROOT::EnableImplicitMT(numberOfThreads);

//     if (trainingFraction <= 0.0 || trainingFraction >= 1.0)
//         throw std::runtime_error("trainingFraction must be between 0 and 1");

//     logLine("--- ML RUN SETTINGS ---");
//     logValue("Use prepared data", UsePreparedData);
//     logValue("Training dataset", TrainingDatasetPath);
//     logValue("Testing dataset", TestingDatasetPath);
//     logValue("Training fraction", trainingFraction);
//     logValue("Testing fraction", 1.0 - trainingFraction);
//     logValue("Development mode", DevelopmentMode);
//     if (DevelopmentMode) logValue("Development files per sample", DevelopmentFilesPerSample);
//     logValue("Maximum muons/event", MaxMuons);
//     logValue("Muons used as features", ModelMuons);
//     logValue("Number of features", features.size());
//     logValue("Training threads", numberOfThreads);

//     logLine("[Features]");
//     for (const auto& feature : features) logValue("Feature", feature);

//     logLine("[BDT model parameters]");
//     logValue("Method", "BDTG");
//     logValue("Number of trees", numberOfTrees);
//     logValue("Maximum depth", maximumDepth);
//     logValue("Minimum node size [%]", minimumNodeSizePercent);
//     logValue("Bagged sample fraction", baggedSampleFraction);
//     logValue("Number of cuts", numberOfCuts);
//     logValue("Shrinkage", shrinkage);
//     logValue("Random seed", randomSeed);
//     logValue("Decision threshold", decisionThreshold);
//     logLine("------");

//     if (!UsePreparedData) {
//         logLine("--- CREATE TRAINING AND TESTING DATA ---");

//         const auto dyFiles = loadFileList(DyFileListPath, DevelopmentMode, DevelopmentFilesPerSample);
//         std::vector<std::string> allFiles = dyFiles;

//         for (const auto& fileList : BackgroundFileListPaths) {
//             const auto files = loadFileList(fileList, DevelopmentMode, DevelopmentFilesPerSample);
//             allFiles.insert(allFiles.end(), files.begin(), files.end());
//         }

//         logValue("DY files", dyFiles.size());
//         logValue("All MC files", allFiles.size());

//         // Robust identification of the main DY sample.
//         auto isMainDySample = [dyFiles](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
//             for (const auto& file : dyFiles)
//                 if (sampleInfo.Contains(file)) return 1;
//             return 0;
//         };

//         // Same event-level logic as filterMuons(), except HLT is applied as a common preselection below.
//         auto passesDyTriggerSelection = [](const RVec<float>& muPt,
//                                            const RVec<float>& muEta,
//                                            const RVec<float>& muPhi,
//                                            const RVec<int>& muCharge,
//                                            const RVec<Bool_t>& muIsGlobal,
//                                            const RVec<float>& trigPt,
//                                            const RVec<float>& trigEta,
//                                            const RVec<float>& trigPhi,
//                                            const RVec<int>& trigId) {
//             const size_t nMu = muPt.size();
//             if (nMu < 2) return false;

//             for (size_t i = 0; i < nMu; ++i) {
//                 if (!muIsGlobal[i] || std::abs(muEta[i]) >= 2.4f) continue;

//                 for (size_t k = i + 1; k < nMu; ++k) {
//                     if (!muIsGlobal[k] || std::abs(muEta[k]) >= 2.4f) continue;
//                     if (muCharge[i] * muCharge[k] >= 0) continue;

//                     const float leadPt = std::max(muPt[i], muPt[k]);
//                     const float subPt = std::min(muPt[i], muPt[k]);

//                     if (leadPt <= 22.f || subPt <= 10.f) continue;

//                     for (size_t j = 0; j < trigPt.size(); ++j) {
//                         if (std::abs(trigId[j]) != 13 || trigPt[j] <= 20.f) continue;

//                         const float dPhi1 = dphi(muPhi[i], trigPhi[j]);
//                         const float dEta1 = muEta[i] - trigEta[j];
//                         const float dPhi2 = dphi(muPhi[k], trigPhi[j]);
//                         const float dEta2 = muEta[k] - trigEta[j];

//                         if (dEta1*dEta1 + dPhi1*dPhi1 < 0.01f ||
//                             dEta2*dEta2 + dPhi2*dPhi2 < 0.01f)
//                             return true;
//                     }
//                 }
//             }

//             return false;
//         };

//         // Same event-level logic as filterMuonQualityAndCosmicRejection().
//         auto passesDyQualitySelection = [](const RVec<float>& pt,
//                                            const RVec<float>& eta,
//                                            const RVec<float>& phi,
//                                            const RVec<int>& nStations,
//                                            const RVec<int>& nTrackerLayers,
//                                            const RVec<float>& dxy,
//                                            const RVec<float>& ptErr,
//                                            const RVec<float>& tkRelIso) {
//             const size_t n = pt.size();

//             for (size_t i = 0; i < n; ++i) {
//                 if (nStations[i] <= 1) continue;
//                 if (nTrackerLayers[i] <= 5) continue;
//                 if (std::abs(dxy[i]) >= 0.2f) continue;
//                 if (pt[i] <= 0.f || ptErr[i] / pt[i] >= 0.3f) continue;
//                 if (tkRelIso[i] >= 0.1f) continue;

//                 for (size_t j = i + 1; j < n; ++j) {
//                     if (nStations[j] <= 1) continue;
//                     if (nTrackerLayers[j] <= 5) continue;
//                     if (std::abs(dxy[j]) >= 0.2f) continue;
//                     if (pt[j] <= 0.f || ptErr[j] / pt[j] >= 0.3f) continue;
//                     if (tkRelIso[j] >= 0.1f) continue;

//                     if (std::abs(3.14159265358979323846f -
//                                  openingAngle3D(eta[i], phi[i], eta[j], phi[j])) <= 0.005f)
//                         continue;

//                     return true;
//                 }
//             }

//             return false;
//         };

//         ROOT::RDataFrame rawMcDf("Events", allFiles);

//         ROOT::RDF::RNode mlDf = rawMcDf
//             .DefinePerSample("isMainDySample", isMainDySample)

//             // Common preselection applied equally to signal and backgrounds.
//             .Filter([MaxMuons](const RVec<float>& pt) {
//                 return pt.size() >= 2 && pt.size() <= MaxMuons;
//             }, {"Muon_pt"})
//             .Filter("HLT_IsoMu24")

//             // Original physics-selection result stored as booleans instead of filtering events away.
//             .Define("PassDyTrigger", passesDyTriggerSelection,
//                 {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_charge", "Muon_isGlobal",
//                  "TrigObj_pt", "TrigObj_eta", "TrigObj_phi", "TrigObj_id"})

//             .Define("PassDyQuality", passesDyQualitySelection,
//                 {"Muon_pt", "Muon_eta", "Muon_phi", "Muon_nStations",
//                  "Muon_nTrackerLayers", "Muon_dxy", "Muon_ptErr", "Muon_tkRelIso"})

//             // Correct label: selected main-DY event = 1, everything else = 0.
//             .Define("IsDyProcess",
//                 [](int isDy, bool passTrigger, bool passQuality) {
//                     return isDy == 1 && passTrigger && passQuality ? 1 : 0;
//                 },
//                 {"isMainDySample", "PassDyTrigger", "PassDyQuality"})

//             .Define("MuonCount",
//                 [](const RVec<float>& pt) { return static_cast<float>(pt.size()); },
//                 {"Muon_pt"})

//             // Stable ordering: Muon0 = highest-pT, Muon1 = second highest-pT.
//             .Define("MlMuonOrder", [](const RVec<float>& pt) {
//                 auto indices = ROOT::VecOps::Argsort(pt);
//                 std::reverse(indices.begin(), indices.end());
//                 return indices;
//             }, {"Muon_pt"});

//         const std::vector<std::pair<std::string, std::string>> rawFields = {
//             {"pt", "Muon_pt"},
//             {"eta", "Muon_eta"},
//             {"phi", "Muon_phi"},
//             {"charge", "Muon_charge"},
//             {"isGlobal", "Muon_isGlobal"},
//             {"nStations", "Muon_nStations"},
//             {"nTrackerLayers", "Muon_nTrackerLayers"},
//             {"dxy", "Muon_dxy"},
//             {"tkRelIso", "Muon_tkRelIso"}
//         };

//         for (size_t i = 0; i < ModelMuons; ++i) {
//             const std::string n = std::to_string(i);
//             const std::string index = "MlMuonOrder[" + n + "]";

//             for (const auto& [name, branch] : rawFields)
//                 mlDf = mlDf.Define(
//                     "Muon" + n + "_" + name,
//                     "static_cast<float>(" + branch + "[" + index + "])"
//                 );

//             mlDf = mlDf.Define(
//                 "Muon" + n + "_ptRelErr",
//                 "Muon_pt[" + index + "] > 0.f ? "
//                 "static_cast<float>(Muon_ptErr[" + index + "] / Muon_pt[" + index + "]) : 999.f"
//             );
//         }

//         // Strong physics variables.
//         mlDf = makeDimuonKinematicsDataFrame(
//             makeLeadingSubleadingMuonKinematicsDataFrame(mlDf)
//         );

//         mlDf = mlDf
//             .Define("Dimuon_deltaPhi",
//                 [](float phi1, float phi2) {
//                     return std::abs(dphi(phi1, phi2));
//                 },
//                 {"LeadingMuon_phi", "SubleadingMuon_phi"})

//             .Define("Dimuon_deltaR",
//                 [](float eta1, float phi1, float eta2, float phi2) {
//                     const float dEta = eta1 - eta2;
//                     const float dPhi = dphi(phi1, phi2);
//                     return std::sqrt(dEta*dEta + dPhi*dPhi);
//                 },
//                 {"LeadingMuon_eta", "LeadingMuon_phi",
//                  "SubleadingMuon_eta", "SubleadingMuon_phi"})

//             .Define("Dimuon_openingAngle3D",
//                 [](float eta1, float phi1, float eta2, float phi2) {
//                     return openingAngle3D(eta1, phi1, eta2, phi2);
//                 },
//                 {"LeadingMuon_eta", "LeadingMuon_phi",
//                  "SubleadingMuon_eta", "SubleadingMuon_phi"})

//             // Stable train/test split key. Do NOT use rdfentry_ here.
//             .Define("mlSplitKey",
//                 [](UInt_t run, UInt_t lumi, ULong64_t event) {
//                     ULong64_t x = event;
//                     x ^= static_cast<ULong64_t>(run) << 32;
//                     x ^= static_cast<ULong64_t>(lumi) << 16;
//                     x ^= x >> 33;
//                     x *= 0xff51afd7ed558ccdULL;
//                     x ^= x >> 33;
//                     x *= 0xc4ceb9fe1a85ec53ULL;
//                     x ^= x >> 33;
//                     return x % 10000ULL;
//                 },
//                 {"run", "luminosityBlock", "event"});

//         const ULong64_t splitModulo = 10000;
//         const ULong64_t trainLimit = static_cast<ULong64_t>(trainingFraction * splitModulo);

//         auto signalBeforeSplit = mlDf.Filter("IsDyProcess == 1").Count();
//         auto backgroundBeforeSplit = mlDf.Filter("IsDyProcess == 0").Count();

//         auto trainingNode = mlDf.Filter(
//             [trainLimit](ULong64_t splitKey) { return splitKey < trainLimit; },
//             {"mlSplitKey"}
//         );

//         auto testingNode = mlDf.Filter(
//             [trainLimit](ULong64_t splitKey) { return splitKey >= trainLimit; },
//             {"mlSplitKey"}
//         );

//         std::vector<std::string> outputColumns = features;
//         outputColumns.insert(outputColumns.end(), {"event", "IsDyProcess"});

//         ROOT::RDF::RSnapshotOptions snapshotOptions;
//         snapshotOptions.fLazy = true;

//         auto trainingSnapshot = trainingNode.Snapshot(
//             "Events", TrainingDatasetPath, outputColumns, snapshotOptions
//         );

//         auto testingSnapshot = testingNode.Snapshot(
//             "Events", TestingDatasetPath, outputColumns, snapshotOptions
//         );

//         std::vector<ROOT::RDF::RResultHandle> preparationHandles;
//         preparationHandles.emplace_back(signalBeforeSplit);
//         preparationHandles.emplace_back(backgroundBeforeSplit);
//         preparationHandles.emplace_back(trainingSnapshot);
//         preparationHandles.emplace_back(testingSnapshot);

//         ROOT::RDF::RunGraphs(preparationHandles);

//         logValue("Signal before split", *signalBeforeSplit);
//         logValue("Background before split", *backgroundBeforeSplit);
//         logElapsedTime("Training/testing files created");
//     }
//     else {
//         logLine("--- LOAD EXISTING TRAINING AND TESTING DATA ---");
//         logElapsedTime("Prepared datasets selected");
//     }

//     ROOT::RDataFrame trainingDf("Events", TrainingDatasetPath);
//     ROOT::RDataFrame testingDf("Events", TestingDatasetPath);

//     auto trainingSignal = trainingDf.Filter("IsDyProcess == 1").Count();
//     auto trainingBackground = trainingDf.Filter("IsDyProcess == 0").Count();

//     const double TrainBaselineAccuracy = static_cast<double>(*trainingSignal) / static_cast<double>(*trainingBackground);
//     logValue(" Training Baseline accuracy (always background)", TrainBaselineAccuracy);

//     auto testingSignal = testingDf.Filter("IsDyProcess == 1").Count();
//     auto testingBackground = testingDf.Filter("IsDyProcess == 0").Count();

//     const double TestBaselineAccuracy = static_cast<double>(*testingSignal) / static_cast<double>(*testingBackground);
//     logValue(" Testing Baseline accuracy (always background)", TestBaselineAccuracy);

//     std::vector<ROOT::RDF::RResultHandle> countHandles;
//     countHandles.emplace_back(trainingSignal);
//     countHandles.emplace_back(trainingBackground);
//     countHandles.emplace_back(testingSignal);
//     countHandles.emplace_back(testingBackground);

//     ROOT::RDF::RunGraphs(countHandles);

//     logValue("Training signal", *trainingSignal);
//     logValue("Training background", *trainingBackground);
//     logValue("Testing signal", *testingSignal);
//     logValue("Testing background", *testingBackground);

//     const std::string weightsFile = trainDrellYanBdt(
//         features,
//         "Events",
//         TrainingDatasetPath,
//         TestingDatasetPath,
//         "MlTemp/TMVA_DY_training.root",
//         "MlTemp/BDT_training_history.root",
//         "DrellYanClassification",
//         "BDTG",
//         "DY_BDT_dataset",
//         numberOfTrees,
//         maximumDepth,
//         minimumNodeSizePercent,
//         baggedSampleFraction,
//         numberOfCuts,
//         shrinkage,
//         randomSeed
//     );

//     logValue("Trained model", weightsFile);
//     logElapsedTime("BDT training END");

//     evaluateDrellYanBdt(
//         features,
//         "Events",
//         TestingDatasetPath,
//         weightsFile,
//         "BDTG",
//         "MlTemp/ML_DY_evaluation.root",
//         "Predictions",
//         decisionThreshold,
//         LOG_FILE
//     );

//     logElapsedTime("BDT evaluation END");
//     logElapsedTime("MlRun complete");
// }




// void MlRun()
// {
//     resetExecutionTimer();
//     resetLog();
//     gSystem->mkdir("MlTemp", true);

//     const unsigned int numberOfThreads = 4;
//     const bool DevelopmentMode = true;
//     const size_t DevelopmentFilesPerSample = 1;
//     const bool UsePreparedData = true; // false = create training/testing ROOT files, true = load them
//     const double trainingFraction = 0.70;

//     const std::string TrainingDatasetPath = "MlTemp/ML_DY_training.root";
//     const std::string TestingDatasetPath = "MlTemp/ML_DY_testing.root";
//     const std::string DyFileListPath = "dy_mc_main_xrootd_files.txt";
//     const std::vector<std::string> BackgroundFileListPaths = {
//         "Background_dataset_files/ttbar_xrootd_files.txt",
//         "Background_dataset_files/single_top_tw_xrootd_files.txt",
//         "Background_dataset_files/single_antitop_tw_xrootd_files.txt",
//         "Background_dataset_files/ww_xrootd_files.txt",
//         "Background_dataset_files/wz_xrootd_files.txt",
//         "Background_dataset_files/zz_xrootd_files.txt",
//         "Background_dataset_files/wjets_xrootd_files.txt",
//         "Background_dataset_files/qcd_pt20to30_xrootd_files.txt"
//     };

//     const size_t MaxMuons = 2;//The dat is cutted up to N muons.

//     const std::vector<std::string> features = [MaxMuons]() {
//         std::vector<std::string> f = {"MuonCount", "HLT_IsoMu24_f", "MET_pt"};
//         const std::vector<std::string> fields = {
//             "pt",
//             // "eta",
//             "phi",
//             "charge",
//             "isGlobal",
//             "nStations",
//             // "nTrackerLayers",
//             // "dxy",
//             // "ptErr",
//             "tkRelIso"
//         };

//         for (size_t i = 0; i < MaxMuons; ++i)
//             for (const std::string& field : fields)
//                 f.push_back("Muon" + std::to_string(i) + "_" + field);

//         return f;
//     }();
//     // const std::vector<std::string> features = {
//     //     "LeadingMuon_pt",
//     //     "SubleadingMuon_pt",
//     //     "LeadingMuon_eta",
//     //     "SubleadingMuon_eta",

//         // "LeadingMuon_tkRelIso",
//         // "SubleadingMuon_tkRelIso",

//         // "LeadingMuon_dxy",
//         // "SubleadingMuon_dxy",

//         // "LeadingMuon_ptRelErr",
//         // "SubleadingMuon_ptRelErr",

//         // "LeadingMuon_nStations",
//         // "SubleadingMuon_nStations",

//     //     "LeadingMuon_nTrackerLayers",
//     //     "SubleadingMuon_nTrackerLayers",

//     //     "Dimuon_mass",
//     //     "Dimuon_pt",
//     //     "Dimuon_rapidity",

//     //     "Dimuon_deltaPhi",
//     //     "Dimuon_deltaR",
//     //     "Dimuon_openingAngle3D",

//     //     "MET_pt"
//     // };

// //HLT_IsoMu24
// //Muons N number 1,2,3...
// //ABS(Muon_eta) 
// //Muon_charge
// // Muon_isGlobal
// //TrigObj_id
// //trigger  pT​
// //trigger ΔR2

//     //Temp Importance
//     // :    1 : Dimuon_mass                   : 2.311e-01
//     // :    2 : Dimuon_deltaPhi               : 1.573e-01
//     // :    3 : Dimuon_deltaR                 : 1.340e-01
//     // :    4 : MET_pt                        : 1.291e-01
//     // :    5 : SubleadingMuon_pt             : 1.127e-01
//     // :    6 : LeadingMuon_tkRelIso          : 6.435e-02
//     // :    7 : SubleadingMuon_nStations      : 4.913e-02
//     // :    8 : Dimuon_openingAngle3D         : 4.375e-02
//     // :    9 : SubleadingMuon_tkRelIso       : 4.153e-02
//     // :   10 : LeadingMuon_nStations         : 1.817e-02
//     // :   11 : LeadingMuon_nTrackerLayers    : 9.807e-03
//     // :   12 : SubleadingMuon_dxy            : 8.624e-03
//     // :   13 : SubleadingMuon_nTrackerLayers : 6.979e-03
//     // :   14 : LeadingMuon_dxy               : 3.200e-03
//     // :   15 : LeadingMuon_eta               : 1.697e-03
//     // :   16 : SubleadingMuon_ptRelErr       : 1.642e-03
//     // :   17 : Dimuon_rapidity               : 1.186e-03
//     // :   18 : SubleadingMuon_eta            : 3.749e-04
//     // :   19 : Dimuon_pt                     : 1.276e-04
//     // :   20 : LeadingMuon_pt                : 1.204e-04
//     // :   21 : LeadingMuon_ptRelErr          : 7.573e-05


//     const unsigned int numberOfTrees = 60;
//     const unsigned int maximumDepth = 6;//A depth-three tree can represent interactions among several variables.
//     const double minimumNodeSizePercent = 0.005;//This controls how small a tree node is allowed to become.
//     const double baggedSampleFraction = 0.3;//It can reduce correlations between trees and decrease overfitting. PVZ Only 20 % is used
//     const unsigned int numberOfCuts = 30;//More cuts can improve split resolution but increase computation. TMVA evaluates a finite set of candidate positions
//     const double shrinkage = 0.05;//It is the gradient boosting learning rate
//     const unsigned int randomSeed = 1;
//     const double decisionThreshold = 0.0;

//     ROOT::EnableImplicitMT(numberOfThreads);
//     if (trainingFraction <= 0.0 || trainingFraction >= 1.0) throw std::runtime_error("trainingFraction must be between 0 and 1");

//     logLine("--- ML RUN SETTINGS ---");
//     logValue("Use prepared data", UsePreparedData);
//     logValue("Training dataset", TrainingDatasetPath);
//     logValue("Testing dataset", TestingDatasetPath);
//     logValue("Training fraction", trainingFraction);
//     logValue("Testing fraction", 1.0 - trainingFraction);
//     logValue("Development mode", DevelopmentMode);
//     if (DevelopmentMode) logValue("Development files per sample", DevelopmentFilesPerSample);
//     logValue("Training threads", numberOfThreads);
//     logLine("[Features]");
//     for (const std::string& feature : features) logValue("Feature", feature);
//     logLine("[BDT model parameters]");
//     logValue("Method", "BDTG"); logValue("Boost type", "Gradient"); logValue("Number of trees", numberOfTrees);
//     logValue("Maximum depth", maximumDepth); logValue("Minimum node size [%]", minimumNodeSizePercent);
//     logValue("Bagged sample fraction", baggedSampleFraction); logValue("Number of cuts", numberOfCuts);
//     logValue("Shrinkage", shrinkage); logValue("Random seed", randomSeed); logValue("Decision threshold", decisionThreshold);
//     logLine("------");

//     if (!UsePreparedData) {
//         logLine("--- CREATE TRAINING AND TESTING DATA ---");

//         const std::vector<std::string> dyFiles = loadFileList(DyFileListPath, DevelopmentMode, DevelopmentFilesPerSample);
//         std::vector<std::string> allFiles = dyFiles;

//         for (const std::string& fileList : BackgroundFileListPaths) {
//             const auto files = loadFileList(fileList, DevelopmentMode, DevelopmentFilesPerSample);
//             allFiles.insert(allFiles.end(), files.begin(), files.end());
//         }

//         logValue("DY files", dyFiles.size());
//         logValue("All MC files", allFiles.size());

//         ROOT::RDataFrame dyDf("Events", dyFiles);
//         const ULong64_t numberOfDyEvents = *dyDf.Count();

//         auto passedResult = FullFilterDF(
//             dyDf.Define("mlEntry", "rdfentry_").Filter([MaxMuons](const RVec<float>& pt) { return pt.size() <= MaxMuons; }, {"Muon_pt"}),
//             "1.0"
//         ).Take<ULong64_t>("mlEntry");

//         const auto passedDyEntries = std::make_shared<std::unordered_set<ULong64_t>>(passedResult->begin(), passedResult->end());

//         logValue("DY events", numberOfDyEvents);
//         logValue("DY events passing FullFilterDF", passedDyEntries->size());

//         ROOT::RDataFrame rawMcDf("Events", allFiles);

//         auto labeledDf = makeMlRawMuonFeaturesDataFrame(rawMcDf.Define("mlEntry", "rdfentry_"), MaxMuons)
//             .Define("IsDyProcess", [passedDyEntries, numberOfDyEvents](ULong64_t entry) {
//                 return entry < numberOfDyEvents && passedDyEntries->find(entry) != passedDyEntries->end() ? 1 : 0;
//             }, {"mlEntry"});

//         const ULong64_t splitModulo = 10000;
//         const ULong64_t trainLimit = static_cast<ULong64_t>(trainingFraction * splitModulo);

//         auto trainingNode = labeledDf.Filter([trainLimit](ULong64_t entry) { return entry % 10000 < trainLimit; }, {"mlEntry"});
//         auto testingNode = labeledDf.Filter([trainLimit](ULong64_t entry) { return entry % 10000 >= trainLimit; }, {"mlEntry"});

//         std::vector<std::string> outputColumns = features;
//         outputColumns.insert(outputColumns.end(), {"event", "IsDyProcess"});

//         std::cout << "Separating muon columns and writing training/testing datasets...\n";
//         ROOT::RDF::Experimental::AddProgressBar(labeledDf);
//         ROOT::RDF::RSnapshotOptions snapshotOptions;
//         snapshotOptions.fLazy = true;

//         auto trainingSnapshot = trainingNode.Snapshot(
//             "Events",
//             TrainingDatasetPath,
//             outputColumns,
//             snapshotOptions
//         );

//         auto testingSnapshot = testingNode.Snapshot(
//             "Events",
//             TestingDatasetPath,
//             outputColumns,
//             snapshotOptions
//         );

//         ROOT::RDF::RunGraphs({trainingSnapshot, testingSnapshot});


//         logElapsedTime("Training/testing files created");
//     } else {
//         logLine("--- LOAD EXISTING TRAINING AND TESTING DATA ---");
//         logElapsedTime("Prepared datasets selected");
//     }

//     ROOT::RDataFrame trainingDf("Events", TrainingDatasetPath);
//     ROOT::RDataFrame testingDf("Events", TestingDatasetPath);
//     auto trainingSignal = trainingDf.Filter("IsDyProcess == 1").Count();
//     auto trainingBackground = trainingDf.Filter("IsDyProcess == 0").Count();
//     auto testingSignal = testingDf.Filter("IsDyProcess == 1").Count();
//     auto testingBackground = testingDf.Filter("IsDyProcess == 0").Count();
//     std::vector<ROOT::RDF::RResultHandle> countHandles;
//     countHandles.emplace_back(trainingSignal); countHandles.emplace_back(trainingBackground); countHandles.emplace_back(testingSignal); countHandles.emplace_back(testingBackground);
//     ROOT::RDF::RunGraphs(countHandles);
//     logValue("Training signal", *trainingSignal); logValue("Training background", *trainingBackground);
//     logValue("Testing signal", *testingSignal); logValue("Testing background", *testingBackground);

//     const std::string weightsFile = trainDrellYanBdt(features, "Events", TrainingDatasetPath, TestingDatasetPath,
//         "MlTemp/TMVA_DY_training.root", "MlTemp/BDT_training_history.root", "DrellYanClassification", "BDTG", "DY_BDT_dataset",
//         numberOfTrees, maximumDepth, minimumNodeSizePercent, baggedSampleFraction, numberOfCuts, shrinkage, randomSeed);

//     logValue("Trained model", weightsFile); logElapsedTime("BDT training END");

//     evaluateDrellYanBdt(features, "Events", TestingDatasetPath, weightsFile, "BDTG", "MlTemp/ML_DY_evaluation.root", "Predictions", decisionThreshold, LOG_FILE);
//     logElapsedTime("BDT evaluation END");
//     logElapsedTime("MlRun complete");
// }




// docker start -ai cmscloud
// root -l -b -q -e '.L filter_dy1.C' -e 'MlRun()'
// void MlRun()
// {
//     resetExecutionTimer();
//     resetLog();

//     gSystem->mkdir("MlTemp", true);

//     const unsigned int numberOfThreads = 6;
//     ROOT::EnableImplicitMT(numberOfThreads);

//     const bool DevelopmentMode = true;
//     const size_t DevelopmentFilesPerSample = 3;

//     const bool UsePreparedData = true;
//     const std::string TrainingDatasetPath = "MlTemp/ML_DY_training.root";
//     const std::string TestingDatasetPath = "MlTemp/ML_DY_testing.root";
//     logValue("Use prepared data", UsePreparedData);
//     logValue("Training dataset", TrainingDatasetPath);
//     logValue("Testing dataset", TestingDatasetPath);


//     logValue("Development mode", DevelopmentMode);
//     if (DevelopmentMode) {
//         logValue("Development files per sample", DevelopmentFilesPerSample);
//     }

//     // "features" selects which columns will be saved into the labelled ML dataset
//     const std::vector<std::string> features = {
//         "LeadingMuon_pt",
//         "SubleadingMuon_pt",
//         "LeadingMuon_eta",
//         "SubleadingMuon_eta",
//         "Dimuon_mass",
//         "Dimuon_pt",
//         "Dimuon_rapidity",
//         "MET_pt"
//     };

//     const unsigned int numberOfTrees = 360;
//     const unsigned int maximumDepth = 6;//A depth-three tree can represent interactions among several variables.
//     const double minimumNodeSizePercent = 0.001;//This controls how small a tree node is allowed to become.
//     const double baggedSampleFraction = 0.7;//It can reduce correlations between trees and decrease overfitting. PVZ Only 20 % is used
//     const unsigned int numberOfCuts = 40;//More cuts can improve split resolution but increase computation. TMVA evaluates a finite set of candidate positions
//     const double shrinkage = 0.05;//It is the gradient boosting learning rate
//     const unsigned int randomSeed = 1;
//     const double decisionThreshold = 0.0;

//     //Precision is especially important for selecting a tiny sample. It is the main metrics TP/(TP+FP).
//     logLine("--- BDT training parameters ---");
//     logLine("[BDT model parameters]");
//     logValue("Method", "BDTG");
//     logValue("Boost type", "Gradient");
//     logValue("Number of trees", numberOfTrees);
//     logValue("Maximum depth", maximumDepth);
//     logValue("Minimum node size [%]", minimumNodeSizePercent);
//     logValue("Bagged sample fraction", baggedSampleFraction);
//     logValue("Number of cuts", numberOfCuts);
//     logValue("Shrinkage", shrinkage);
//     logValue("Random seed", randomSeed);
//     logValue("Decision threshold", decisionThreshold);
//     logValue("Number of features", features.size());
//     logValue("Training threads", numberOfThreads);
//     logLine("---");

    
//     std::optional<ROOT::RDF::RNode> trainingDf;
//     std::optional<ROOT::RDF::RNode> testingDf;

//     if (!UsePreparedData) {
//         logLine("--- DATA FILTERING AND NORMALIZATION ---");
//         std::vector<std::string> allFiles;
//         std::map<std::string, int> sampleIdByFile;
//         std::map<std::string, double> normalizationByFile;

//         /////Normalization parralel
//         using SampleConfig = std::tuple<int, std::string, std::string, double>;
//         const std::vector<SampleConfig> samples = {
//             {0, "main_dy", "dy_mc_main_xrootd_files.txt", 5396.0},
//             {1, "ttbar", "Background_dataset_files/ttbar_xrootd_files.txt", 687.1},
//             {2, "single_top_tw", "Background_dataset_files/single_top_tw_xrootd_files.txt", 32.45},
//             {3, "single_antitop_tw", "Background_dataset_files/single_antitop_tw_xrootd_files.txt", 32.51},
//             {4, "ww", "Background_dataset_files/ww_xrootd_files.txt", 75.88},
//             {5, "wz", "Background_dataset_files/wz_xrootd_files.txt", 27.56},
//             {6, "zz", "Background_dataset_files/zz_xrootd_files.txt", 12.08},
//             {7, "wjets", "Background_dataset_files/wjets_xrootd_files.txt", 54110.0},
//             {8, "qcd_pt20to30", "Background_dataset_files/qcd_pt20to30_xrootd_files.txt", 2.529e6}
//         };

//         std::vector<std::vector<std::string>> filesBySample;
//         std::vector<std::unique_ptr<ROOT::RDataFrame>> runsDataFrames;
//         std::vector<ROOT::RDF::RResultPtr<double>> sumGenWeights;
//         std::vector<ROOT::RDF::RResultHandle> normalizationHandles;

//         filesBySample.reserve(samples.size());
//         runsDataFrames.reserve(samples.size());
//         sumGenWeights.reserve(samples.size());
//         normalizationHandles.reserve(samples.size());

//         // Book all normalization calculations without running them yet.
//         for (const auto& sample : samples) {
//             const auto& fileList = std::get<2>(sample);
//             filesBySample.push_back(loadFileList(fileList, DevelopmentMode, DevelopmentFilesPerSample));
//             runsDataFrames.push_back(std::make_unique<ROOT::RDataFrame>("Runs", filesBySample.back()));
//             sumGenWeights.push_back(runsDataFrames.back()->Sum<double>("genEventSumw"));
//             normalizationHandles.emplace_back(sumGenWeights.back());
//         }

//         // Run the independent normalization graphs concurrently.
//         ROOT::RDF::RunGraphs(normalizationHandles);

//         // Collect the results after parallel execution.
//         for (size_t i = 0; i < samples.size(); ++i) {
//             const int sampleId = std::get<0>(samples[i]);
//             const std::string& sampleName = std::get<1>(samples[i]);
//             const double crossSectionPb = std::get<3>(samples[i]);
//             const double sumGenWeight = *sumGenWeights[i];

//             if (sumGenWeight == 0.0) {throw std::runtime_error("Zero genEventSumw for sample: " + sampleName);}

//             // Integrated luminosity is 1.0 fb^-1 here.
//             const double normalization =crossSectionPb * 1000.0 / sumGenWeight;

//             for (const std::string& file : filesBySample[i]) {
//                 sampleIdByFile[file] = sampleId;
//                 normalizationByFile[file] = normalization;
//                 allFiles.push_back(file);
//             }

//             logValue(sampleName + " sum gen weights", sumGenWeight);
//             logValue(sampleName + " normalization", normalization);
//         }

//         /////////////

//         logElapsedTime("MC file loading and normalization END");

//         auto findSampleId =[sampleIdByFile](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
//             for (const auto& item : sampleIdByFile)
//                 if (sampleInfo.Contains(item.first))
//                     return item.second;

//             throw std::runtime_error("No sample ID for: " + sampleInfo.AsString());
//         };

//         auto findNormalization =[normalizationByFile](unsigned int, const ROOT::RDF::RSampleInfo& sampleInfo) {
//                 for (const auto& item : normalizationByFile)
//                     if (sampleInfo.Contains(item.first))
//                         return item.second;

//                 throw std::runtime_error("No normalization for: " + sampleInfo.AsString());
//             };

//         ROOT::RDataFrame rawMcDf("Events", allFiles);

//         auto mcDf = rawMcDf
//             .Define("mlEntry", "rdfentry_")//creates a new column named mlEntry containing the event’s sequential entry number in the input RDataFrame rdfentry_ is a special ROOT column: 0, 1, 2, 3, ... mlEntry stores a copy under a normal column name.
//             .DefinePerSample("mcSampleId", findSampleId)
//             .DefinePerSample("sampleNormalization", findNormalization);

//         auto mcKinematicsDf =
//             makeDimuonKinematicsDataFrame(
//                 makeLeadingSubleadingMuonKinematicsDataFrame(
//                     mcDf
//                 )
//             );

//         auto mcInputDf = mcKinematicsDf
//             .Define(
//                 "eventWeight",
//                 "static_cast<double>(genWeight) * "
//                 "sampleNormalization"
//             )
//             .Define(
//                 "mlWeight",
//                 "std::abs(eventWeight)"
//             );

//         logElapsedTime("Muon and dimuon input columns END");

//         auto passedDyEntriesResult = filterMuonQualityAndCosmicRejection(filterMuons(mcInputDf.Filter("mcSampleId == 0"))).Take<ULong64_t>("mlEntry"); //mcSampleId == 0 means only the main MC Dy sample.

//         const auto passedDyEntries =
//             std::make_shared<
//                 std::unordered_set<ULong64_t>
//             >(
//                 passedDyEntriesResult->begin(),
//                 passedDyEntriesResult->end()
//             );

//         logValue("Main DY events passing original filters", passedDyEntries->size());

//         logElapsedTime("Original DY filter labeling END");

//         //Assigning the 0/1 labels for training.
//         auto labeledMcDf = mcInputDf.Define("IsDyProcess",
//             [](int sampleId) { return sampleId == 0 ? 1 : 0; },
//             {"mcSampleId"});

//         std::vector<std::string> labeledColumns = features;

//         labeledColumns.insert(
//             labeledColumns.end(),
//             {
//                 "mlEntry",
//                 "event",
//                 "mcSampleId",
//                 "IsDyProcess",
//                 "eventWeight",
//                 "mlWeight"
//             }
//         );

//         labeledMcDf.Snapshot("Events", "MlTemp/ML_DY_labeled.root", labeledColumns);//Save the selected columns dataframe and saves it.

//         logElapsedTime("Labeled dataset snapshot END");

//         ROOT::RDataFrame labeledDf("Events", "MlTemp/ML_DY_labeled.root");

//         trainingDf.emplace(labeledDf.Filter("mlEntry % 10 < 7"));
//         testingDf.emplace(labeledDf.Filter("mlEntry % 10 >= 7"));
//         auto signalCount =labeledDf.Filter("IsDyProcess == 1").Count();
//         auto backgroundCount = labeledDf.Filter("IsDyProcess == 0").Count();
//         auto trainingCount = (*trainingDf).Count();
//         auto testingCount = (*testingDf).Count();

//         auto signalWeight =labeledDf.Filter("IsDyProcess == 1").Sum<double>("mlWeight");
//         auto backgroundWeight =labeledDf.Filter("IsDyProcess == 0").Sum<double>("mlWeight");

//         std::vector<ROOT::RDF::RResultHandle> handles;

//         handles.emplace_back(signalCount);
//         handles.emplace_back(backgroundCount);
//         handles.emplace_back(trainingCount);
//         handles.emplace_back(testingCount);
//         handles.emplace_back(signalWeight);
//         handles.emplace_back(backgroundWeight);

//         ROOT::RDF::RunGraphs(handles);

//         logValue("Signal events", *signalCount);
//         logValue("Background events", *backgroundCount);
//         logValue("Training events", *trainingCount);
//         logValue("Testing events", *testingCount);
//         logValue("Normalized signal weight", *signalWeight);
//         logValue("Normalized background weight", *backgroundWeight);

//         logElapsedTime("Dataset counting and splitting END");
//         logLine("------");
//     } else{
//         logLine("--- EXISTING DATA LOADING ---");

//         ROOT::RDataFrame preparedTrainingDf("Events", TrainingDatasetPath);
//         ROOT::RDataFrame preparedTestingDf("Events", TestingDatasetPath);

//         trainingDf.emplace(preparedTrainingDf);
//         testingDf.emplace(preparedTestingDf);

//         logElapsedTime("Prepared datasets loaded");
//         logLine("------");
//     }

//     if (!trainingDf.has_value() || !testingDf.has_value()) {
//         throw std::runtime_error(
//             "Training or testing RDataFrame was not initialized"
//         );
//     }
//     logLine("--- FEATURES ---");
//     for (const std::string& feature : features) {
//         logValue("Feature", feature);
//     }
//     logLine("------");
//     // createFeatureMatrixMGL(*trainingDf, features);

//     // Baseline: predict every testing event as negative/background.
//     // const double totalWeight =*(*testingDf).Sum<double>("mlWeight");
//     // const double negativeWeight = *(*testingDf).Filter("IsDyProcess == 0").Sum<double>("mlWeight");
//     // logValue("Baseline accuracy (always negative)", totalWeight > 0.0 ? negativeWeight / totalWeight : 0.0);

//     const std::string weightsFile =
//     trainDrellYanBdt(
//         *trainingDf,
//         *testingDf,
//         UsePreparedData,
//         features,
//         "Events",
//         "MlTemp/ML_DY_training.root",
//         "MlTemp/ML_DY_testing.root",
//         "MlTemp/TMVA_DY_training.root",

//         "MlTemp/BDT_training_history.root",

//         "DrellYanClassification",
//         "BDTG",
//         "DY_BDT_dataset",
//         numberOfTrees,
//         maximumDepth,
//         minimumNodeSizePercent,
//         baggedSampleFraction,
//         numberOfCuts,
//         shrinkage,
//         randomSeed
//     );

//     logValue("Trained model", weightsFile);

//     logElapsedTime("BDT training END");

//     evaluateDrellYanBdt(
//         *testingDf,
//         UsePreparedData,
//         features,
//         "Events",
//         "MlTemp/ML_DY_testing.root",
//         weightsFile,
//         "BDTG",
//         "MlTemp/ML_DY_evaluation.root",
//         "Predictions",
//         decisionThreshold,
//         LOG_FILE                            
//     );

//     logElapsedTime("BDT evaluation END");
//     logElapsedTime("MlRun complete");
// }