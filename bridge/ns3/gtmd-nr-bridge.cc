// GTMD-RL <-> ns-3 5G-LENA socket bridge scenario.
//
// This is the ns-3 side of the near-RT-RIC control loop described in the paper.
// A Python server (bridge/rl_allocation_server.py) owns the DSIC mechanism and the
// epoch-frozen RL policy; once per epoch it ships a per-slice PRB budget over a TCP
// socket. This program is the gNB/RIC scenario: it receives that budget, enforces
// it on a 3-slice 5G NR downlink (URLLC / eMBB / mMTC) by capping each slice's
// offered rate to what its PRB entitlement can carry, measures per-slice throughput
// and delay with FlowMonitor, and ships the KPIs back. The Python side folds those
// KPIs into the RL reward -- a genuine closed loop, over plain POSIX sockets, with
// no ns3-ai middleware.
//
// Wire protocol (newline-delimited JSON), see bridge/protocol.py:
//   ns3   -> {"type":"hello","role":"ns3"}
//   server-> {"type":"config", ...}
//   loop: server-> {"type":"alloc","epoch":k,"prbs":[...],"prb_capacity_bits":[...]}
//         ns3   -> {"type":"kpi","epoch":k,"throughput_mbps":[...],"mean_delay_ms":[...],...}
//   server-> {"type":"done"}
//
// Build: copy this file into <ns-3>/scratch/ and run
//   ./ns3 build gtmd-nr-bridge
//   ./ns3 run "gtmd-nr-bridge --serverPort=5005"
// (start the Python server first; see bridge/README.md).

#include "ns3/antenna-module.h"
#include "ns3/applications-module.h"
#include "ns3/core-module.h"
#include "ns3/flow-monitor-module.h"
#include "ns3/internet-module.h"
#include "ns3/mobility-module.h"
#include "ns3/nr-module.h"
#include "ns3/point-to-point-module.h"

#include <arpa/inet.h>
#include <cerrno>
#include <cstring>
#include <netinet/in.h>
#include <sstream>
#include <string>
#include <sys/socket.h>
#include <unistd.h>
#include <vector>

using namespace ns3;

NS_LOG_COMPONENT_DEFINE("GtmdNrBridge");

// --------------------------------------------------------------------------- //
// Minimal TCP NDJSON client (real OS socket, blocking).
// --------------------------------------------------------------------------- //
class JsonSocket
{
  public:
    bool Connect(const std::string& host, uint16_t port)
    {
        m_fd = socket(AF_INET, SOCK_STREAM, 0);
        if (m_fd < 0)
        {
            return false;
        }
        sockaddr_in addr{};
        addr.sin_family = AF_INET;
        addr.sin_port = htons(port);
        inet_pton(AF_INET, host.c_str(), &addr.sin_addr);
        return ::connect(m_fd, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) == 0;
    }

    void SendLine(const std::string& s)
    {
        std::string line = s + "\n";
        ::send(m_fd, line.data(), line.size(), 0);
    }

    // Blocking read of one '\n'-terminated line; empty string on EOF.
    std::string RecvLine()
    {
        size_t nl;
        while ((nl = m_buf.find('\n')) == std::string::npos)
        {
            char tmp[8192];
            ssize_t n = ::recv(m_fd, tmp, sizeof(tmp), 0);
            if (n <= 0)
            {
                return "";
            }
            m_buf.append(tmp, static_cast<size_t>(n));
        }
        std::string line = m_buf.substr(0, nl);
        m_buf.erase(0, nl + 1);
        return line;
    }

    void Close()
    {
        if (m_fd >= 0)
        {
            ::close(m_fd);
            m_fd = -1;
        }
    }

  private:
    int m_fd{-1};
    std::string m_buf;
};

// --------------------------------------------------------------------------- //
// Tiny JSON field extractors (numbers and number-arrays by key). Enough for the
// fixed schema above; avoids any external JSON dependency.
// --------------------------------------------------------------------------- //
static std::string
FindValue(const std::string& json, const std::string& key)
{
    std::string pat = "\"" + key + "\"";
    size_t p = json.find(pat);
    if (p == std::string::npos)
    {
        return "";
    }
    p = json.find(':', p + pat.size());
    if (p == std::string::npos)
    {
        return "";
    }
    return json.substr(p + 1);
}

static std::string
JsonString(const std::string& json, const std::string& key)
{
    std::string v = FindValue(json, key);
    size_t a = v.find('"');
    if (a == std::string::npos)
    {
        return "";
    }
    size_t b = v.find('"', a + 1);
    return v.substr(a + 1, b - a - 1);
}

static double
JsonNumber(const std::string& json, const std::string& key)
{
    std::string v = FindValue(json, key);
    return std::strtod(v.c_str(), nullptr);
}

static std::vector<double>
JsonArray(const std::string& json, const std::string& key)
{
    std::vector<double> out;
    std::string v = FindValue(json, key);
    size_t a = v.find('[');
    size_t b = v.find(']', a);
    if (a == std::string::npos || b == std::string::npos)
    {
        return out;
    }
    std::stringstream ss(v.substr(a + 1, b - a - 1));
    std::string tok;
    while (std::getline(ss, tok, ','))
    {
        if (!tok.empty())
        {
            out.push_back(std::strtod(tok.c_str(), nullptr));
        }
    }
    return out;
}

// --------------------------------------------------------------------------- //
// Global run state shared across the scheduled epoch callbacks.
// --------------------------------------------------------------------------- //
struct BridgeState
{
    JsonSocket sock;
    uint32_t nSlices{3};
    uint32_t uePerSlice{2};
    uint32_t totalPrbs{50};
    double epochMs{60.0};
    std::vector<double> slaMs;
    std::vector<std::string> sliceNames;
    // per-slice OnOff apps (downlink), grouped
    std::vector<ApplicationContainer> sliceClients;
    std::vector<uint16_t> slicePortBase; // dest-port base per slice for classifier mapping
    Ptr<FlowMonitor> monitor;
    Ptr<Ipv4FlowClassifier> classifier;
    // FlowMonitor snapshot taken at the start of each epoch (per flow id)
    std::map<FlowId, uint64_t> lastRxBytes;
    std::map<FlowId, double> lastDelaySum; // seconds
    std::map<FlowId, uint64_t> lastRxPackets;
    uint32_t epochsDone{0};
};

static BridgeState g;

// Map a flow to a slice index using the destination port band.
static int
SliceOfFlow(const Ipv4FlowClassifier::FiveTuple& t)
{
    for (uint32_t s = 0; s < g.nSlices; ++s)
    {
        uint16_t base = g.slicePortBase[s];
        if (t.destinationPort >= base && t.destinationPort < base + 100)
        {
            return static_cast<int>(s);
        }
    }
    return -1;
}

// Apply the per-slice PRB budget by rate-capping each slice's offered load to
// budget_i * capacity_per_prb_i (bits/slot) converted to bits/s.
static void
ApplyBudget(const std::vector<double>& prbs, const std::vector<double>& capBits, double slotMs)
{
    for (uint32_t s = 0; s < g.nSlices && s < prbs.size(); ++s)
    {
        double bitsPerSlot = prbs[s] * (s < capBits.size() ? capBits[s] : 0.0);
        double bitsPerSec = bitsPerSlot / (slotMs / 1000.0);
        uint64_t perUe = static_cast<uint64_t>(bitsPerSec / std::max<uint32_t>(g.uePerSlice, 1));
        for (uint32_t u = 0; u < g.sliceClients[s].GetN(); ++u)
        {
            Ptr<OnOffApplication> app = DynamicCast<OnOffApplication>(g.sliceClients[s].Get(u));
            if (app)
            {
                app->SetAttribute("DataRate", DataRateValue(DataRate(perUe)));
            }
        }
    }
}

static void EndEpoch(uint32_t epoch);

// Fired at the start of each epoch: pull the next allocation and program the rates.
static void
RunEpoch(uint32_t epoch)
{
    std::string msg = g.sock.RecvLine();
    if (msg.empty())
    {
        NS_LOG_UNCOND("[ns3] server closed the connection");
        Simulator::Stop();
        return;
    }
    std::string type = JsonString(msg, "type");
    if (type == "done")
    {
        NS_LOG_UNCOND("[ns3] received DONE after " << g.epochsDone << " epochs");
        Simulator::Stop();
        return;
    }
    if (type != "alloc")
    {
        NS_LOG_UNCOND("[ns3] unexpected message type '" << type << "'");
        Simulator::Stop();
        return;
    }
    std::vector<double> prbs = JsonArray(msg, "prbs");
    std::vector<double> capBits = JsonArray(msg, "prb_capacity_bits");
    ApplyBudget(prbs, capBits, 1.0); // slot = 1 ms

    // Snapshot FlowMonitor so EndEpoch can compute this epoch's delta.
    g.monitor->CheckForLostPackets();
    for (const auto& kv : g.monitor->GetFlowStats())
    {
        g.lastRxBytes[kv.first] = kv.second.rxBytes;
        g.lastDelaySum[kv.first] = kv.second.delaySum.GetSeconds();
        g.lastRxPackets[kv.first] = kv.second.rxPackets;
    }

    std::ostringstream b;
    b << "[ns3] epoch " << epoch << " budget=[";
    for (size_t i = 0; i < prbs.size(); ++i)
    {
        b << (int)prbs[i] << (i + 1 < prbs.size() ? "," : "");
    }
    b << "]";
    NS_LOG_UNCOND(b.str());

    Simulator::Schedule(MilliSeconds(g.epochMs), &EndEpoch, epoch);
}

// Fired at the end of each epoch: measure per-slice KPIs and ship them back.
static void
EndEpoch(uint32_t epoch)
{
    g.monitor->CheckForLostPackets();
    std::vector<double> rxBits(g.nSlices, 0.0);
    std::vector<double> delaySum(g.nSlices, 0.0);
    std::vector<double> rxPkts(g.nSlices, 0.0);

    for (const auto& kv : g.monitor->GetFlowStats())
    {
        FlowId id = kv.first;
        Ipv4FlowClassifier::FiveTuple t = g.classifier->FindFlow(id);
        int s = SliceOfFlow(t);
        if (s < 0)
        {
            continue;
        }
        double dBytes = kv.second.rxBytes - (g.lastRxBytes.count(id) ? g.lastRxBytes[id] : 0);
        double dDelay = kv.second.delaySum.GetSeconds() - (g.lastDelaySum.count(id) ? g.lastDelaySum[id] : 0.0);
        double dPkts = kv.second.rxPackets - (g.lastRxPackets.count(id) ? g.lastRxPackets[id] : 0);
        rxBits[s] += dBytes * 8.0;
        delaySum[s] += dDelay;
        rxPkts[s] += dPkts;
    }

    double dur = g.epochMs / 1000.0;
    std::ostringstream out;
    out << "{\"type\":\"kpi\",\"epoch\":" << epoch << ",\"throughput_mbps\":[";
    for (uint32_t s = 0; s < g.nSlices; ++s)
    {
        out << (rxBits[s] / dur / 1e6) << (s + 1 < g.nSlices ? "," : "");
    }
    out << "],\"mean_delay_ms\":[";
    std::vector<double> meanDelay(g.nSlices, 0.0);
    for (uint32_t s = 0; s < g.nSlices; ++s)
    {
        meanDelay[s] = rxPkts[s] > 0 ? 1000.0 * delaySum[s] / rxPkts[s] : 0.0;
        out << meanDelay[s] << (s + 1 < g.nSlices ? "," : "");
    }
    out << "],\"sla_violation\":[";
    for (uint32_t s = 0; s < g.nSlices; ++s)
    {
        double sla = s < g.slaMs.size() ? g.slaMs[s] : 1e9;
        double viol = (meanDelay[s] > sla) ? std::min(1.0, (meanDelay[s] - sla) / sla) : 0.0;
        out << viol << (s + 1 < g.nSlices ? "," : "");
    }
    out << "],\"prb_used\":[";
    for (uint32_t s = 0; s < g.nSlices; ++s)
    {
        out << 0 << (s + 1 < g.nSlices ? "," : "");
    }
    out << "]}";
    g.sock.SendLine(out.str());
    g.epochsDone++;

    // Immediately pull the next epoch's allocation.
    Simulator::ScheduleNow(&RunEpoch, epoch + 1);
}

int
main(int argc, char* argv[])
{
    std::string serverHost = "127.0.0.1";
    uint16_t serverPort = 5005;
    uint16_t numerology = 0; // 1 ms slot, matching the Python model's slot grid
    double centralFrequency = 3.5e9;
    double bandwidth = 20e6;
    double totalTxPower = 43;

    CommandLine cmd;
    cmd.AddValue("serverHost", "RL+DSIC server host", serverHost);
    cmd.AddValue("serverPort", "RL+DSIC server TCP port", serverPort);
    cmd.AddValue("numerology", "NR numerology", numerology);
    cmd.AddValue("centralFrequency", "carrier frequency (Hz)", centralFrequency);
    cmd.AddValue("bandwidth", "channel bandwidth (Hz)", bandwidth);
    cmd.AddValue("totalTxPower", "gNB tx power (dBm)", totalTxPower);
    cmd.Parse(argc, argv);

    // ----------------------------------------------------------------------- //
    // 1. Connect to the RL+DSIC server and complete the handshake.
    // ----------------------------------------------------------------------- //
    if (!g.sock.Connect(serverHost, serverPort))
    {
        NS_FATAL_ERROR("Could not connect to RL server at " << serverHost << ":" << serverPort
                                                            << " (start bridge/rl_allocation_server.py first)");
    }
    g.sock.SendLine("{\"type\":\"hello\",\"role\":\"ns3\",\"version\":1}");
    std::string cfg = g.sock.RecvLine();
    NS_ABORT_MSG_IF(cfg.empty(), "no CONFIG from server");
    g.nSlices = static_cast<uint32_t>(JsonNumber(cfg, "total_prbs") > 0 ? 3 : 3);
    g.totalPrbs = static_cast<uint32_t>(JsonNumber(cfg, "total_prbs"));
    g.epochMs = JsonNumber(cfg, "epoch_length") * JsonNumber(cfg, "slot_ms");
    g.slaMs = JsonArray(cfg, "sla_latency_ms");
    g.nSlices = g.slaMs.size() ? static_cast<uint32_t>(g.slaMs.size()) : 3;
    uint32_t epochs = static_cast<uint32_t>(JsonNumber(cfg, "epochs"));
    g.sliceNames = {"URLLC", "eMBB", "mMTC"};
    NS_LOG_UNCOND("[ns3] CONFIG: slices=" << g.nSlices << " B=" << g.totalPrbs
                                          << " epochMs=" << g.epochMs << " epochs=" << epochs);

    // ----------------------------------------------------------------------- //
    // 2. Build the 5G-NR scenario: 1 gNB, uePerSlice UEs per slice, 1 BWP, OFDMA QoS.
    // ----------------------------------------------------------------------- //
    Config::SetDefault("ns3::NrRlcUm::MaxTxBufferSize", UintegerValue(999999999));
    uint32_t totalUe = g.nSlices * g.uePerSlice;

    GridScenarioHelper grid;
    grid.SetRows(1);
    grid.SetColumns(1);
    grid.SetHorizontalBsDistance(5.0);
    grid.SetBsHeight(10.0);
    grid.SetUtHeight(1.5);
    grid.SetSectorization(GridScenarioHelper::SINGLE);
    grid.SetBsNumber(1);
    grid.SetUtNumber(totalUe);
    grid.SetScenarioHeight(20);
    grid.SetScenarioLength(20);
    grid.CreateScenario();

    Ptr<NrPointToPointEpcHelper> epc = CreateObject<NrPointToPointEpcHelper>();
    Ptr<IdealBeamformingHelper> bf = CreateObject<IdealBeamformingHelper>();
    Ptr<NrHelper> nr = CreateObject<NrHelper>();
    nr->SetBeamformingHelper(bf);
    nr->SetEpcHelper(epc);
    epc->SetAttribute("S1uLinkDelay", TimeValue(MilliSeconds(0)));

    // OFDMA QoS scheduler: honours slice priority; the PRB budget is realised by
    // rate-capping each slice below (see ApplyBudget). See bridge/README.md for the
    // custom weighted-scheduler variant.
    nr->SetSchedulerTypeId(TypeId::LookupByName("ns3::NrMacSchedulerOfdmaQos"));

    std::string errorModel = "ns3::NrEesmIrT2";
    nr->SetDlErrorModel(errorModel);
    nr->SetUlErrorModel(errorModel);
    nr->SetGnbDlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    nr->SetGnbUlAmcAttribute("AmcModel", EnumValue(NrAmc::ErrorModel));
    bf->SetAttribute("BeamformingMethod", TypeIdValue(DirectPathBeamforming::GetTypeId()));

    nr->SetUeAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetUeAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetUeAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));
    nr->SetGnbAntennaAttribute("NumRows", UintegerValue(1));
    nr->SetGnbAntennaAttribute("NumColumns", UintegerValue(1));
    nr->SetGnbAntennaAttribute("AntennaElement", PointerValue(CreateObject<IsotropicAntennaModel>()));

    CcBwpCreator ccBwpCreator;
    CcBwpCreator::SimpleOperationBandConf bandConf(centralFrequency, bandwidth, 1);
    bandConf.m_numBwp = 1;
    OperationBandInfo band = ccBwpCreator.CreateOperationBandContiguousCc(bandConf);
    Ptr<NrChannelHelper> channelHelper = CreateObject<NrChannelHelper>();
    channelHelper->ConfigureFactories("UMi", "Default", "ThreeGpp");
    channelHelper->SetPathlossAttribute("ShadowingEnabled", BooleanValue(false));
    channelHelper->AssignChannelsToBands({band});
    BandwidthPartInfoPtrVector allBwps = CcBwpCreator::GetAllBwps({band});

    // Route all QCIs to the single BWP.
    nr->SetGnbBwpManagerAlgorithmAttribute("NGBR_LOW_LAT_EMBB", UintegerValue(0));
    nr->SetGnbBwpManagerAlgorithmAttribute("GBR_CONV_VOICE", UintegerValue(0));
    nr->SetGnbBwpManagerAlgorithmAttribute("NGBR_VIDEO_TCP_DEFAULT", UintegerValue(0));

    NodeContainer gnbNodes = grid.GetBaseStations();
    NodeContainer ueNodes = grid.GetUserTerminals();

    NetDeviceContainer gnbDev = nr->InstallGnbDevice(gnbNodes, allBwps);
    NetDeviceContainer ueDev = nr->InstallUeDevice(ueNodes, allBwps);
    int64_t stream = 1;
    stream += nr->AssignStreams(gnbDev, stream);
    stream += nr->AssignStreams(ueDev, stream);

    NrHelper::GetGnbPhy(gnbDev.Get(0), 0)->SetAttribute("Numerology", UintegerValue(numerology));
    double x = pow(10, totalTxPower / 10);
    NrHelper::GetGnbPhy(gnbDev.Get(0), 0)->SetAttribute("TxPower", DoubleValue(10 * log10(x)));

    auto [remoteHost, remoteHostAddr] = epc->SetupRemoteHost("100Gb/s", 2500, Seconds(0.0));
    InternetStackHelper internet;
    internet.Install(ueNodes);
    Ipv4InterfaceContainer ueIp = epc->AssignUeIpv4Address(NetDeviceContainer(ueDev));
    nr->AttachToClosestGnb(ueDev, gnbDev);

    // ----------------------------------------------------------------------- //
    // 3. Per-slice downlink OnOff apps (rate reconfigured every epoch).
    // ----------------------------------------------------------------------- //
    g.slicePortBase = {2000, 2100, 2200};
    ApplicationContainer serverApps;
    g.sliceClients.assign(g.nSlices, ApplicationContainer());

    for (uint32_t u = 0; u < totalUe; ++u)
    {
        uint32_t s = u / g.uePerSlice; // slice index of this UE
        uint16_t port = g.slicePortBase[s] + (u % g.uePerSlice);

        PacketSinkHelper sink("ns3::UdpSocketFactory",
                              InetSocketAddress(Ipv4Address::GetAny(), port));
        serverApps.Add(sink.Install(ueNodes.Get(u)));

        OnOffHelper onoff("ns3::UdpSocketFactory",
                          InetSocketAddress(ueIp.GetAddress(u), port));
        onoff.SetAttribute("OnTime", StringValue("ns3::ConstantRandomVariable[Constant=1000]"));
        onoff.SetAttribute("OffTime", StringValue("ns3::ConstantRandomVariable[Constant=0]"));
        onoff.SetAttribute("PacketSize", UintegerValue(s == 0 ? 200 : 1200));
        onoff.SetAttribute("DataRate", DataRateValue(DataRate("1Mbps"))); // reprogrammed per epoch
        ApplicationContainer c = onoff.Install(remoteHost);
        g.sliceClients[s].Add(c);

        // Dedicated bearer per slice QCI.
        NrEpsBearer::Qci qci = (s == 0) ? NrEpsBearer::GBR_CONV_VOICE
                                        : (s == 1) ? NrEpsBearer::NGBR_LOW_LAT_EMBB
                                                   : NrEpsBearer::NGBR_VIDEO_TCP_DEFAULT;
        NrEpsBearer bearer(qci);
        Ptr<NrEpcTft> tft = Create<NrEpcTft>();
        NrEpcTft::PacketFilter pf;
        pf.localPortStart = port;
        pf.localPortEnd = port;
        tft->Add(pf);
        nr->ActivateDedicatedEpsBearer(ueDev.Get(u), bearer, tft);
    }

    serverApps.Start(MilliSeconds(1));
    serverApps.Stop(MilliSeconds(10) + MilliSeconds(g.epochMs * (epochs + 2)));
    for (auto& sc : g.sliceClients)
    {
        sc.Start(MilliSeconds(10));
        sc.Stop(MilliSeconds(10) + MilliSeconds(g.epochMs * (epochs + 2)));
    }

    FlowMonitorHelper fmHelper;
    NodeContainer endpoints;
    endpoints.Add(remoteHost);
    endpoints.Add(ueNodes);
    g.monitor = fmHelper.Install(endpoints);
    g.monitor->SetAttribute("DelayBinWidth", DoubleValue(0.0001));

    // Kick off the epoch loop just after the apps start.
    Simulator::Schedule(MilliSeconds(11), &RunEpoch, 0u);

    // classifier is only valid after Install; fetch once here.
    g.classifier = DynamicCast<Ipv4FlowClassifier>(fmHelper.GetClassifier());

    Simulator::Stop(MilliSeconds(20) + MilliSeconds(g.epochMs * (epochs + 3)));
    Simulator::Run();

    NS_LOG_UNCOND("[ns3] finished; epochs exchanged = " << g.epochsDone);
    g.sock.Close();
    Simulator::Destroy();
    return 0;
}
