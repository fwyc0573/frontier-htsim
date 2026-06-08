// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "config.h"
#include <sstream>

#include <iostream>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fstream>
#include <cctype>
#include "network.h"
#include "randomqueue.h"
#include "shortflows.h"
#include "pipe.h"
#include "eventlist.h"
#include "logfile.h"
#include "loggers.h"
#include "clock.h"
#include "ndp.h"
#include "compositequeue.h"
#include "firstfit.h"
#include "topology.h"
#include "queue_lossless_input.h"
#include "connection_matrix.h"

#include "fat_tree_topology.h"
#include "fat_tree_switch.h"
#include "rail_topology.h"
#include "rail_ocs_config.h"
#include "rail_ocs_schedule.h"
#include "rail_ocs_topology.h"
#include "rail3_config.h"
#include "rail3_topology.h"
#include "rail3_ftlike_topology.h"
#include "rail_ocs3_config.h"
#include "rail_ocs3_port_schedule.h"
#include "rail_ocs3_topology.h"
#include "mixnet_config.h"
#include "mixnet_topology.h"
#include "ep_group_ocs_topology.h"

#include <list>

// Simulation params

//#define PRINTPATHS 1

#define PERIODIC 0
#include "main.h"

uint32_t RTT = 1; // this is per link delay in us; identical RTT microseconds = 0.02 ms
int DEFAULT_NODES = 432;
#define DEFAULT_QUEUE_SIZE 15

//#define SWITCH_BUFFER (SERVICE * RTT / 1000)
#define USE_FIRST_FIT 0
#define FIRST_FIT_INTERVAL 100

EventList eventlist;

void exit_error(char* progr) {
    cout << "Usage " << progr << " [-nodes N]\n\t[-conns C]\n\t[-cwnd cwnd_size]\n\t[-q queue_size]\n\t[-oversubscribed_cc] Use receiver-driven AIMD to reduce total window when trims are not last hop\n\t[-queue_type composite|random|lossless|lossless_input|]\n\t[-tm traffic_matrix_file]\n\t[-topo fat_tree_topo_file]\n\t[-rail servers spines [gpus_per_server]]\n\t[-rail3_topo rail3_topo_file]\n\t[-ocs_topo rail_ocs_topo_file]\n\t[-ocs3_topo rail_ocs3_topo_file]\n\t[-ocs3_dump_reach output_file]\n\t[-mixnet_topo mixnet_topo_file]\n\t[-strat route_strategy (single,rand,perm,pull,ecmp,\n\tecmp_host path_count,ecmp_ar,ecmp_rr,\n\tecmp_host_ar ar_thresh)]\n\t[-log log_level]\n\t[-seed random_seed]\n\t[-end end_time_in_usec]\n\t[-mtu MTU]\n\t[-hop_latency x] per hop wire latency in us,default 1\n\t[-switch_latency x] switching latency in us, default 0\n\t[-host_queue_type  swift|prio|fair_prio]" << endl;
    exit(1);
}

void filter_paths(uint32_t src_id, vector<const Route*>& paths, FatTreeTopology* top) {
    uint32_t num_servers = top->no_of_servers();
    uint32_t num_cores = top->no_of_cores();
    uint32_t num_pods = top->no_of_pods();
    uint32_t pod_switches = top->tor_switches_per_pod();

    uint32_t path_classes = pod_switches/2;
    cout << "srv: " << num_servers << " cores: " << num_cores << " pods: " << num_pods << " pod_sw: " << pod_switches << " classes: " << path_classes << endl;
    uint32_t pclass = src_id % path_classes;
    cout << "src: " << src_id << " class: " << pclass << endl;

    for (uint32_t r = 0; r < paths.size(); r++) {
        const Route* rt = paths.at(r);
        if (rt->size() == 12) {
            BaseQueue* q = dynamic_cast<BaseQueue*>(rt->at(6));
            cout << "Q:" << atoi(q->str().c_str()+2) << " " << q->str() << endl;
            uint32_t core = atoi(q->str().c_str()+2);
            if (core % path_classes != pclass) {
                paths[r] = NULL;
            }
        }
    }
}

int main(int argc, char **argv) {
    Clock c(timeFromSec(5 / 100.), eventlist);
    mem_b queuesize = DEFAULT_QUEUE_SIZE;
    linkspeed_bps linkspeed = speedFromMbps((double)HOST_NIC);
    int packet_size = 9000;
    uint32_t path_entropy_size = 10000000;
    uint32_t no_of_conns = 0, cwnd = 15, no_of_nodes = DEFAULT_NODES;
    uint32_t tiers = 3; // we support 2 and 3 tier fattrees
    double logtime = 0.25; // ms;
    stringstream filename(ios_base::out);
    simtime_picosec hop_latency = timeFromUs((uint32_t)1);
    simtime_picosec switch_latency = timeFromUs((uint32_t)0);
    queue_type qt = COMPOSITE;

    bool log_sink = false;
    bool rts = false;
    bool log_tor_downqueue = false;
    bool log_tor_upqueue = false;
    bool log_traffic = false;
    bool log_switches = false;
    bool log_queue_usage = false;
    double ecn_thresh = 0.5; // default marking threshold for ECN load balancing
    RouteStrategy route_strategy = NOT_SET;
    int seed = 13;
    int path_burst = 1;
    int i = 1;

    bool oversubscribed_congestion_control = false;

    filename << "logout.dat";
    int end_time = 1000;//in microseconds

    queue_type snd_type = FAIR_PRIO;

    float ar_sticky_delta = 10;
    FatTreeSwitch::sticky_choices ar_sticky = FatTreeSwitch::PER_PACKET;
    uint64_t high_pfc = 15, low_pfc = 12;

    char* tm_file = NULL;
    char* topo_file = NULL;
    char* ocs_topo_file = NULL;
    char* rail3_topo_file = NULL;
    char* ocs3_topo_file = NULL;
    char* mixnet_topo_file = NULL;
    char* ocs3_dump_reach_file = NULL;
    bool use_rail = false;
    uint32_t rail_servers = 0;
    uint32_t rail_gpus_per_server = 8;
    uint32_t rail_spines = 0;
    uint32_t rail_servers_per_tor = 0; // optional: cap hosts per ToR per rail (enables rail split)

    while (i<argc) {
        if (!strcmp(argv[i],"-o")) {
            filename.str(std::string());
            filename << argv[i+1];
            i++;
        } else if (!strcmp(argv[i],"-oversubscribed_cc")) {
            oversubscribed_congestion_control = true;
        } else if (!strcmp(argv[i],"-conns")) {
            no_of_conns = atoi(argv[i+1]);
            cout << "no_of_conns "<<no_of_conns << endl;
            i++;
        } else if (!strcmp(argv[i],"-end")) {
            end_time = atoi(argv[i+1]);
            cout << "endtime(us) "<< end_time << endl;
            i++;            
        } else if (!strcmp(argv[i],"-rts")) {
            rts = true;
            cout << "rts enabled "<< endl;
        } else if (!strcmp(argv[i],"-nodes")) {
            no_of_nodes = atoi(argv[i+1]);
            cout << "no_of_nodes "<<no_of_nodes << endl;
            i++;
        } else if (!strcmp(argv[i],"-tiers")) {
            tiers = atoi(argv[i+1]);
            cout << "tiers "<< tiers << endl;
            assert(tiers == 2 || tiers == 3);
            i++;
        } else if (!strcmp(argv[i],"-queue_type")) {
            if (!strcmp(argv[i+1], "composite")) {
                qt = COMPOSITE;
            } 
            else if (!strcmp(argv[i+1], "composite_ecn")) {
                qt = COMPOSITE_ECN;
            }
            else if (!strcmp(argv[i+1], "lossless")) {
                qt = LOSSLESS;
            }
            else if (!strcmp(argv[i+1], "lossless_input")) {
                qt = LOSSLESS_INPUT;
            }
            else {
                cout << "Unknown queue type " << argv[i+1] << endl;
                exit_error(argv[0]);
            }
            cout << "queue_type "<< qt << endl;
            i++;
        } else if (!strcmp(argv[i],"-host_queue_type")) {
            if (!strcmp(argv[i+1], "swift")) {
                snd_type = SWIFT_SCHEDULER;
            } 
            else if (!strcmp(argv[i+1], "prio")) {
                snd_type = PRIORITY;
            }
            else if (!strcmp(argv[i+1], "fair_prio")) {
                snd_type = FAIR_PRIO;
            }
            else {
                cout << "Unknown host queue type " << argv[i+1] << " expecting one of swift|prio|fair_prio" << endl;
                exit_error(argv[0]);
            }
            cout << "host queue_type "<< snd_type << endl;
            i++;
        } else if (!strcmp(argv[i],"-log")){
            if (!strcmp(argv[i+1], "sink")) {
                log_sink = true;
            } else if (!strcmp(argv[i+1], "sink")) {
                cout << "logging sinks\n";
                log_sink = true;
            } else if (!strcmp(argv[i+1], "tor_downqueue")) {
                cout << "logging tor downqueues\n";
                log_tor_downqueue = true;
            } else if (!strcmp(argv[i+1], "tor_upqueue")) {
                cout << "logging tor upqueues\n";
                log_tor_upqueue = true;
            } else if (!strcmp(argv[i+1], "switch")) {
                cout << "logging total switch queues\n";
                log_switches = true;
            } else if (!strcmp(argv[i+1], "traffic")) {
                cout << "logging traffic\n";
                log_traffic = true;
            } else if (!strcmp(argv[i+1], "queue_usage")) {
                cout << "logging queue usage\n";
                log_queue_usage = true;
            } else {
                exit_error(argv[0]);
            }
            i++;
        } else if (!strcmp(argv[i],"-cwnd")) {
            cwnd = atoi(argv[i+1]);
            cout << "cwnd "<< cwnd << endl;
            i++;
        } else if (!strcmp(argv[i],"-tm")){
            tm_file = argv[i+1];
            cout << "traffic matrix input file: "<< tm_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-topo")){
            topo_file = argv[i+1];
            cout << "FatTree topology input file: "<< topo_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-ocs_topo")) {
            ocs_topo_file = argv[i+1];
            cout << "RailOCS topology input file: " << ocs_topo_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-rail3_topo")) {
            rail3_topo_file = argv[i+1];
            cout << "Rail3 topology input file: " << rail3_topo_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-ocs3_topo")) {
            ocs3_topo_file = argv[i+1];
            cout << "RailOCS3 topology input file: " << ocs3_topo_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-mixnet_topo")) {
            mixnet_topo_file = argv[i+1];
            cout << "MixNet topology input file: " << mixnet_topo_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-ocs3_dump_reach")) {
            ocs3_dump_reach_file = argv[i+1];
            cout << "RailOCS3 reachability dump file: " << ocs3_dump_reach_file << endl;
            i++;
        } else if (!strcmp(argv[i],"-rail")) {
            // Rail topology: -rail <servers> <spines> [gpus_per_server] [servers_per_tor]
            use_rail = true;
            rail_servers = (uint32_t)atoi(argv[i+1]);
            rail_spines = (uint32_t)atoi(argv[i+2]);
            if (i + 3 < argc && argv[i+3][0] != '-') {
                rail_gpus_per_server = (uint32_t)atoi(argv[i+3]);
                i++;
            }
            if (i + 3 < argc && argv[i+3][0] != '-') {
                rail_servers_per_tor = (uint32_t)atoi(argv[i+3]);
                i++;
            }
            i += 2;
            cout << "rail topology enabled: servers=" << rail_servers
                 << " spines=" << rail_spines
                 << " gpus_per_server=" << rail_gpus_per_server;
            if (rail_servers_per_tor) cout << " servers_per_tor=" << rail_servers_per_tor;
            cout << endl;
        } else if (!strcmp(argv[i],"-q")){
            queuesize = atoi(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i],"-ecn_thresh")){
            // fraction of queuesize, between 0 and 1
            ecn_thresh = atof(argv[i+1]); 
            i++;
        } else if (!strcmp(argv[i],"-logtime")){
            logtime = atof(argv[i+1]);            
            cout << "logtime "<< logtime << " ms" << endl;
            i++;
        } else if (!strcmp(argv[i],"-linkspeed")){
            // linkspeed specified is in Mbps
            linkspeed = speedFromMbps(atof(argv[i+1]));
            i++;
        } else if (!strcmp(argv[i],"-seed")){
            seed = atoi(argv[i+1]);
            cout << "random seed "<< seed << endl;
            i++;
        } else if (!strcmp(argv[i],"-mtu")){
            packet_size = atoi(argv[i+1]);
            i++;
        } else if (!strcmp(argv[i],"-paths")){
            path_entropy_size = atoi(argv[i+1]);
            cout << "no of paths " << path_entropy_size << endl;
            i++;
        } else if (!strcmp(argv[i],"-path_burst")){
            path_burst = atoi(argv[i+1]);
            cout << "path burst " << path_burst << endl;
            i++;
        } else if (!strcmp(argv[i],"-hop_latency")){
            hop_latency = timeFromUs(atof(argv[i+1]));
            cout << "Hop latency set to " << timeAsUs(hop_latency) << endl;
            i++;
        } else if (!strcmp(argv[i],"-switch_latency")){
            switch_latency = timeFromUs(atof(argv[i+1]));
            cout << "Switch latency set to " << timeAsUs(switch_latency) << endl;
            i++;
        } else if (!strcmp(argv[i],"-ar_sticky_delta")){
            ar_sticky_delta = atof(argv[i+1]);
            cout << "Adaptive routing sticky delta " << ar_sticky_delta << "us" << endl;
            i++;
        } else if (!strcmp(argv[i],"-pfc_thresholds")){
            low_pfc = atoi(argv[i+1]);
            high_pfc = atoi(argv[i+2]);
            cout << "PFC thresholds high " << high_pfc << " low " << low_pfc << endl;
            i++;
        } else if (!strcmp(argv[i],"-ar_granularity")){
            if (!strcmp(argv[i+1],"packet"))
                ar_sticky = FatTreeSwitch::PER_PACKET;
            else if (!strcmp(argv[i+1],"flow"))
                ar_sticky = FatTreeSwitch::PER_FLOWLET;
            else  {
                cout << "Expecting -ar_granularity packet|flow, found " << argv[i+1] << endl;
                exit(1);
            }   
            i++;
        } else if (!strcmp(argv[i],"-ar_method")){
            if (!strcmp(argv[i+1],"pause")){
                cout << "Adaptive routing based on pause state " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pause;
            }
            else if (!strcmp(argv[i+1],"queue")){
                cout << "Adaptive routing based on queue size " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_queuesize;
            }
            else if (!strcmp(argv[i+1],"bandwidth")){
                cout << "Adaptive routing based on bandwidth utilization " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_bandwidth;
            }
            else if (!strcmp(argv[i+1],"pqb")){
                cout << "Adaptive routing based on pause, queuesize and bandwidth utilization " << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pqb;
            }
            else if (!strcmp(argv[i+1],"pq")){
                cout << "Adaptive routing based on pause, queuesize" << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pq;
            }
            else if (!strcmp(argv[i+1],"pb")){
                cout << "Adaptive routing based on pause, bandwidth utilization" << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_pb;
            }
            else if (!strcmp(argv[i+1],"qb")){
                cout << "Adaptive routing based on queuesize, bandwidth utilization" << endl;
                FatTreeSwitch::fn = &FatTreeSwitch::compare_qb; 
            }
            else {
                cout << "Unknown AR method expecting one of pause, queue, bandwidth, pqb, pq, pb, qb" << endl;
                exit(1);
            }
            i++;
        } else if (!strcmp(argv[i],"-strat")){
            if (!strcmp(argv[i+1], "perm")) {
                route_strategy = SCATTER_PERMUTE;
            } else if (!strcmp(argv[i+1], "rand")) {
                route_strategy = SCATTER_RANDOM;
            } else if (!strcmp(argv[i+1], "ecmp")) {
                route_strategy = SCATTER_ECMP;
            } else if (!strcmp(argv[i+1], "pull")) {
                route_strategy = PULL_BASED;
            } else if (!strcmp(argv[i+1], "single")) {
                route_strategy = SINGLE_PATH;
            } else if (!strcmp(argv[i+1], "ecmp_host")) {
                route_strategy = ECMP_FIB;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
            } else if (!strcmp(argv[i+1], "rr_ecmp")) {
                //this is the host route strategy;
                route_strategy = ECMP_FIB_ECN;
                qt = COMPOSITE_ECN_LB;
                //this is the switch route strategy. 
                FatTreeSwitch::set_strategy(FatTreeSwitch::RR_ECMP);
            } else if (!strcmp(argv[i+1], "ecmp_host_ecn")) {
                route_strategy = ECMP_FIB_ECN;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
                qt = COMPOSITE_ECN_LB;
            } else if (!strcmp(argv[i+1], "reactive_ecn")) {
                // Jitu's suggestion for something really simple
                // One path at a time, but switch whenever we get a trim or ecn
                //this is the host route strategy;
                route_strategy = REACTIVE_ECN;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP);
                qt = COMPOSITE_ECN_LB;
            } else if (!strcmp(argv[i+1], "ecmp_ar")) {
                route_strategy = ECMP_FIB;
                path_entropy_size = 1;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ADAPTIVE_ROUTING);
            } else if (!strcmp(argv[i+1], "ecmp_host_ar")) {
                route_strategy = ECMP_FIB;
                FatTreeSwitch::set_strategy(FatTreeSwitch::ECMP_ADAPTIVE);
                //the stuff below obsolete
                //FatTreeSwitch::set_ar_fraction(atoi(argv[i+2]));
                //cout << "AR fraction: " << atoi(argv[i+2]) << endl;
                //i++;
            } else if (!strcmp(argv[i+1], "ecmp_rr")) {
                // switch round robin
                route_strategy = ECMP_FIB;
                path_entropy_size = 1;
                FatTreeSwitch::set_strategy(FatTreeSwitch::RR);
            }
            i++;
        } else {
            cout << "Unknown parameter " << argv[i] << endl;
            exit_error(argv[0]);
        }
                
        i++;
    }

    srand(seed);
    srandom(seed);
    cout << "Parsed args\n";
    Packet::set_packet_size(packet_size);

    NdpSink::_oversubscribed_congestion_control = oversubscribed_congestion_control;

    if (oversubscribed_congestion_control)
        cout << "Using oversubscribed congestion control " << endl;

    FatTreeSwitch::_ar_sticky = ar_sticky;
    FatTreeSwitch::_sticky_delta = timeFromUs(ar_sticky_delta);
    FatTreeSwitch::_ecn_threshold_fraction = ecn_thresh;

    LosslessInputQueue::_high_threshold = Packet::data_packet_size()*high_pfc;
    LosslessInputQueue::_low_threshold = Packet::data_packet_size()*low_pfc;


    eventlist.setEndtime(timeFromUs((uint32_t)end_time));
    queuesize = memFromPkt(queuesize);
    
    switch (route_strategy) {
    case ECMP_FIB_ECN:
    case REACTIVE_ECN:
        if (qt != COMPOSITE_ECN_LB) {
            fprintf(stderr, "Route Strategy is ECMP ECN.  Must use an ECN queue\n");
            exit(1);
        }
        if (ecn_thresh <= 0 || ecn_thresh >= 1) {
            fprintf(stderr, "Route Strategy is ECMP ECN.  ecn_thresh must be between 0 and 1\n");
            exit(1);
        }
        // no break, fall through
    case ECMP_FIB:
    case SCATTER_ECMP:
        if (path_entropy_size > 10000) {
            fprintf(stderr, "Route Strategy is ECMP.  Must specify path count using -paths\n");
            exit(1);
        }
        break;
    case SINGLE_PATH:
        if (path_entropy_size < 10000 && path_entropy_size > 1) {
            fprintf(stderr, "Route Strategy is SINGLE_PATH, but multiple paths are specifiec using -paths\n");
            exit(1);
        }
        break;
    case NOT_SET:
        fprintf(stderr, "Route Strategy not set.  Use the -strat param.  \nValid values are perm, rand, pull, rg and single\n");
        exit(1);
    default:
        break;
    }

    // prepare the loggers

    cout << "Logging to " << filename.str() << endl;
    //Logfile 
    Logfile logfile(filename.str(), eventlist);

    cout << "Linkspeed set to " << linkspeed/1000000000 << "Gbps" << endl;
    logfile.setStartTime(timeFromSec(0));

    NdpSinkLoggerSampling sinkLogger = NdpSinkLoggerSampling(timeFromMs(logtime), eventlist);
    if (log_sink) {
        logfile.addLogger(sinkLogger);
    }
    NdpTrafficLogger traffic_logger = NdpTrafficLogger();
    if (log_traffic) {
        logfile.addLogger(traffic_logger);
    }

#if PRINT_PATHS
    filename << ".paths";
    cout << "Logging path choices to " << filename.str() << endl;
    std::ofstream paths(filename.str().c_str());
    if (!paths){
        cout << "Can't open for writing paths file!"<<endl;
        exit(1);
    }
#endif

    NdpSrc::setMinRTO(50000); //increase RTO to avoid spurious retransmits
    NdpSrc::setPathEntropySize(path_entropy_size);
    NdpSrc::setRouteStrategy(route_strategy);
    NdpSink::setRouteStrategy(route_strategy);

    NdpSrc* ndpSrc;
    NdpSink* ndpSnk;

    Route* routeout, *routein;

    // scanner interval must be less than min RTO
    NdpRtxTimerScanner ndpRtxScanner(timeFromUs((uint32_t)9), eventlist);
   
    QueueLoggerFactory *qlf = 0;
    if (log_tor_downqueue || log_tor_upqueue) {
        qlf = new QueueLoggerFactory(&logfile, QueueLoggerFactory::LOGGER_SAMPLING, eventlist);
        qlf->set_sample_period(timeFromUs(10.0));
    } else if (log_queue_usage) {
        qlf = new QueueLoggerFactory(&logfile, QueueLoggerFactory::LOGGER_EMPTY, eventlist);
        qlf->set_sample_period(timeFromUs(10.0));
    }
#ifdef FAT_TREE
    Topology* top = nullptr;
    if (rail3_topo_file) {
        Rail3Config cfg;
        if (!load_rail3_config(rail3_topo_file, cfg)) {
            cout << "Failed to load Rail3 topo file: " << rail3_topo_file << endl;
            exit(1);
        }
        uint32_t rail_nodes = cfg.pods * cfg.podsize;
        if (no_of_nodes == (uint32_t)DEFAULT_NODES) {
            no_of_nodes = rail_nodes;
        }
        if (no_of_nodes != rail_nodes) {
            cout << "Error: -nodes (" << no_of_nodes << ") must equal Pods*Podsize ("
                 << rail_nodes << ") for Rail3 topology." << endl;
            exit(1);
        }

        if (cfg.link_speed_gbps > 0.0) {
            linkspeed = speedFromGbps(cfg.link_speed_gbps);
        }
        if (cfg.link_latency_ns > 0) {
            hop_latency = timeFromNs(cfg.link_latency_ns);
        }
        if (cfg.switch_latency_ns > 0) {
            switch_latency = timeFromNs(cfg.switch_latency_ns);
        }

        cout << "Rail3 topo loaded: pods=" << cfg.pods
             << " podsize=" << cfg.podsize
             << " gpus_per_server=" << cfg.gpus_per_server
             << " servers_per_tor=" << cfg.servers_per_tor
             << " tor_up=" << cfg.tor_up
             << " agg_count=" << (cfg.agg_count ? cfg.agg_count : cfg.tor_up)
             << " agg_up=" << cfg.agg_up
             << (cfg.link_speed_gbps > 0.0 ? (" linkspeed_gbps=" + std::to_string(cfg.link_speed_gbps)) : "")
             << endl;

        if (cfg.mode == Rail3Config::RAIL3_FTLIKE) {
            cout << "Rail3FTLike enabled (Type Rail3FT): aligning semantics with FatTree3 composite behavior\n";
            top = new Rail3FtLikeTopology(
                cfg,
                linkspeed,
                queuesize,
                hop_latency,
                switch_latency,
                qt,
                snd_type,
                &logfile,
                &eventlist
            );
        } else {
            top = new Rail3Topology(
                cfg,
                linkspeed,
                queuesize,
                hop_latency,
                switch_latency,
                qt,
                snd_type,
                &logfile,
                &eventlist
            );
        }
    } else if (ocs3_topo_file) {
        RailOcs3Config cfg;
        if (!load_rail_ocs3_config(ocs3_topo_file, cfg)) {
            cout << "Failed to load RailOCS3 topo file: " << ocs3_topo_file << endl;
            exit(1);
        }
        uint32_t rail_nodes = cfg.nodes();
        if (no_of_nodes == (uint32_t)DEFAULT_NODES) {
            no_of_nodes = rail_nodes;
        }
        if (no_of_nodes != rail_nodes) {
            cout << "Error: -nodes (" << no_of_nodes << ") must equal Pods*Podsize ("
                 << rail_nodes << ") for RailOCS3 topology." << endl;
            exit(1);
        }
        // Static OCS assumption: use slot 0 connectivity, no reconfig.
        linkspeed = speedFromGbps(cfg.link_speed_gbps);
        hop_latency = timeFromNs(cfg.link_latency_ns);
        switch_latency = timeFromNs(cfg.switch_latency_ns);
        simtime_picosec ocs_hop_latency = timeFromNs(cfg.ocs_link_latency_ns ? cfg.ocs_link_latency_ns : cfg.link_latency_ns);
        simtime_picosec ocs_switch_latency = timeFromNs(cfg.ocs_switch_latency_ns ? cfg.ocs_switch_latency_ns : cfg.switch_latency_ns);

        cout << "RailOCS3 topo loaded: pods=" << cfg.pods
             << " podsize=" << cfg.podsize
             << " gpus_per_server=" << cfg.gpus_per_server
             << " tor_up=" << cfg.tor_up
             << " trunk_ports_per_pod=" << cfg.trunk_ports_per_pod
             << " agg_planes=" << cfg.agg_ocs_planes
             << " core_planes=" << cfg.core_ocs_planes
             << " linkspeed_gbps=" << cfg.link_speed_gbps
             << " ocs_queue_pkts=" << cfg.ocs_queue_pkts
             << " ocs_no_drop=" << (cfg.ocs_no_drop ? 1 : 0)
             << " ocs_cut_through=" << (cfg.ocs_cut_through ? 1 : 0)
             << " host_feeder_buffer=" << (cfg.host_feeder_buffer ? 1 : 0)
             << " slot_us=" << cfg.slot_us
             << " reconfig_us=" << cfg.reconfig_us
             << endl;

        // Load schedules (slot 0 only).
        RailOcs3PortSchedule agg_sched;
        if (!load_rail_ocs3_port_schedule_slot0(cfg.agg_ocs_schedule_file, cfg.agg_ocs_ports_per_pod(), cfg.agg_ocs_planes, agg_sched, true)) {
            cout << "Failed to load RailOCS3 Agg-OCS schedule file: " << cfg.agg_ocs_schedule_file << endl;
            exit(1);
        }
        RailOcs3PortSchedule core_sched;
        if (!load_rail_ocs3_port_schedule_slot0(cfg.core_ocs_schedule_file, cfg.core_ocs_ports_total(), cfg.core_ocs_planes, core_sched, true)) {
            cout << "Failed to load RailOCS3 Core-OCS schedule file: " << cfg.core_ocs_schedule_file << endl;
            exit(1);
        }

        mem_b ocs_queuesize = queuesize;
        if (cfg.ocs_queue_pkts > 0) {
            ocs_queuesize = memFromPkt(cfg.ocs_queue_pkts);
        }

        top = new RailOcs3Topology(
            cfg,
            agg_sched,
            core_sched,
            linkspeed,
            queuesize,
            ocs_queuesize,
            hop_latency,
            switch_latency,
            ocs_hop_latency,
            ocs_switch_latency,
            cfg.ocs_no_drop,
            qt,
            snd_type,
            &logfile,
            &eventlist
        );

        if (ocs3_dump_reach_file) {
            auto* topo3 = dynamic_cast<RailOcs3Topology*>(top);
            if (topo3) {
                topo3->dump_reachability(std::string(ocs3_dump_reach_file), 200, 2048);
            } else {
                cout << "WARNING: -ocs3_dump_reach specified but topology is not RailOcs3Topology.\n";
            }
        }
    } else if (mixnet_topo_file) {
        MixNetConfig mixcfg;
        if (!load_mixnet_config(mixnet_topo_file, mixcfg)) {
            cout << "Failed to load MixNet topo file: " << mixnet_topo_file << endl;
            exit(1);
        }
        if (no_of_nodes == (uint32_t)DEFAULT_NODES) {
            cout << "Error: MixNet requires explicit -nodes (default is " << DEFAULT_NODES << ")." << endl;
            exit(1);
        }

        // Load EP-group map: supports either:
        // - "node group local_port" per line (recommended; local_port should be ep_idx)
        // - "node group" per line (legacy; local_port inferred by insertion order within group)
        std::vector<uint32_t> ep_group_id;
        std::vector<uint32_t> ep_local_port;
        ep_group_id.resize(no_of_nodes, 0);
        ep_local_port.resize(no_of_nodes, 0);
        {
            std::ifstream f(mixcfg.ep_group_map_file);
            if (!f.is_open()) {
                cout << "Failed to open MixNet EpGroupMapFile: " << mixcfg.ep_group_map_file << endl;
                exit(1);
            }
            std::string ln;
            uint32_t line_idx = 0;
            // group -> next local port (legacy mode)
            std::vector<uint32_t> legacy_next_lp;
            while (std::getline(f, ln)) {
                if (ln.empty() || ln[0] == '#') continue;
                std::stringstream ss(ln);
                uint32_t a = 0, b = 0, c = 0;
                if (!(ss >> a)) continue;
                if (!(ss >> b)) {
                    cout << "MixNet EpGroupMapFile: invalid line (need at least 2 fields): " << ln << endl;
                    exit(1);
                }
                if (a >= no_of_nodes) {
                    cout << "MixNet EpGroupMapFile: node id out of range: " << a << endl;
                    exit(1);
                }
                ep_group_id[a] = b;
                if (ss >> c) {
                    ep_local_port[a] = c;
                } else {
                    if (legacy_next_lp.size() <= b) legacy_next_lp.resize(b + 1, 0);
                    ep_local_port[a] = legacy_next_lp[b]++;
                    (void)line_idx;
                }
            }
        }

        // EPS: fat-tree loaded from a standard FatTreeTopology .topo file.
        // For MixNet(B), EPS topology only contains gateway GPUs:
        // eps_nodes = servers * num_gateways_per_server.
        Topology* eps_top = FatTreeTopology::load(mixcfg.eps_topo_file.c_str(), qlf, eventlist, queuesize, qt, snd_type);
        if (!eps_top) {
            cout << "MixNet: failed to load EPS topo file: " << mixcfg.eps_topo_file << endl;
            exit(1);
        }
        if (no_of_nodes % mixcfg.gpus_per_server != 0) {
            cout << "MixNet: requires -nodes divisible by GpusPerServer\n";
            exit(1);
        }
        uint32_t mix_servers = (uint32_t)no_of_nodes / mixcfg.gpus_per_server;
        uint32_t gw_per_server = (uint32_t)mixcfg.eps_gateway_local_indices.size();
        uint32_t eps_expected_nodes = mix_servers * gw_per_server;
        if (eps_top->no_of_nodes() != eps_expected_nodes) {
            cout << "MixNet: EPS topo node count mismatch: eps.no_of_nodes=" << eps_top->no_of_nodes()
                 << " but expected servers*gw_per_server=" << eps_expected_nodes << endl;
            exit(1);
        }

        // OCS: EP-group local OCS crossbar with static slot0 matching.
        linkspeed_bps ocs_linkspeed = (mixcfg.ocs_link_speed_gbps > 0.0) ? speedFromGbps(mixcfg.ocs_link_speed_gbps) : linkspeed;
        simtime_picosec ocs_hop_lat = (mixcfg.ocs_link_latency_ns > 0) ? timeFromNs(mixcfg.ocs_link_latency_ns) : hop_latency;
        simtime_picosec ocs_sw_lat = (mixcfg.ocs_switch_latency_ns > 0) ? timeFromNs(mixcfg.ocs_switch_latency_ns) : switch_latency;
        mem_b ocs_queuesize = queuesize;
        if (mixcfg.ocs_queue_pkts > 0) {
            ocs_queuesize = memFromPkt(mixcfg.ocs_queue_pkts);
        }

        // Infer group_size as max(local_port)+1 and validate schedule tors matches.
        uint32_t group_size = 0;
        for (uint32_t n = 0; n < no_of_nodes; n++) {
            if (ep_local_port[n] + 1 > group_size) group_size = ep_local_port[n] + 1;
        }
        if (group_size == 0) {
            cout << "MixNet: invalid EP local_port mapping (group_size=0)\n";
            exit(1);
        }
        RailOcsStaticSchedule sched;
        if (!load_rail_ocs_static_schedule_slot0(mixcfg.ocs_schedule_file, group_size, mixcfg.ocs_planes, sched)) {
            cout << "MixNet: failed to load OCS schedule file: " << mixcfg.ocs_schedule_file << endl;
            exit(1);
        }

        Topology* ocs_top = new EpGroupOcsTopology(
            ep_group_id,
            ep_local_port,
            group_size,
            sched,
            ocs_linkspeed,
            queuesize,
            ocs_queuesize,
            hop_latency,
            switch_latency,
            ocs_hop_lat,
            ocs_sw_lat,
            mixcfg.ocs_no_drop,
            snd_type,
            &logfile,
            &eventlist
        );

        // Intra-server gateway forwarding (reachability only): set latency=0 and very high bw.
        // This enforces "only gateway NICs inject to EPS" without modeling its cost yet.
        linkspeed_bps intra_bw = speedFromGbps(100000.0); // 100 Tbps
        mem_b intra_q = memFromPkt(1024);
        simtime_picosec intra_lat = timeFromNs(0);

        top = new MixNetTopology(
            eps_top,
            ocs_top,
            ep_group_id,
            mixcfg.gpus_per_server,
            mixcfg.eps_gateway_local_indices,
            intra_bw,
            intra_q,
            intra_lat,
            snd_type,
            &logfile,
            &eventlist
        );
    } else if (ocs_topo_file) {
        RailOcsConfig cfg;
        if (!load_rail_ocs_config(ocs_topo_file, cfg)) {
            cout << "Failed to load RailOCS topo file: " << ocs_topo_file << endl;
            exit(1);
        }
        uint32_t rail_nodes = cfg.nodes();
        if (no_of_nodes == (uint32_t)DEFAULT_NODES) {
            no_of_nodes = rail_nodes;
        }
        if (no_of_nodes != rail_nodes) {
            cout << "Error: -nodes (" << no_of_nodes << ") must equal Servers*GpusPerServer ("
                 << rail_nodes << ") for RailOCS topology." << endl;
            exit(1);
        }
        // Static OCS assumption: use slot 0 connectivity, no reconfig.
        linkspeed = speedFromGbps(cfg.link_speed_gbps);
        hop_latency = timeFromNs(cfg.link_latency_ns);
        switch_latency = timeFromNs(cfg.switch_latency_ns);
        simtime_picosec ocs_hop_latency = timeFromNs(cfg.ocs_link_latency_ns ? cfg.ocs_link_latency_ns : cfg.link_latency_ns);
        simtime_picosec ocs_switch_latency = timeFromNs(cfg.ocs_switch_latency_ns ? cfg.ocs_switch_latency_ns : cfg.switch_latency_ns);

        use_rail = true;
        rail_servers = cfg.servers;
        rail_spines = cfg.planes; // reuse as number of parallel planes/paths
        rail_gpus_per_server = cfg.gpus_per_server;
        rail_servers_per_tor = cfg.servers_per_tor;
        cout << "RailOCS topo loaded: servers=" << rail_servers
             << " gpus_per_server=" << rail_gpus_per_server
             << " planes=" << rail_spines
             << " linkspeed_gbps=" << cfg.link_speed_gbps
             << " ocs_queue_pkts=" << cfg.ocs_queue_pkts
             << " ocs_no_drop=" << (cfg.ocs_no_drop ? 1 : 0)
             << " ocs_queue_type=" << (cfg.ocs_queue_type.empty() ? "fifo" : cfg.ocs_queue_type)
             << " ocs_cut_through=" << (cfg.ocs_cut_through ? 1 : 0)
             << " slot_us=" << cfg.slot_us
             << " reconfig_us=" << cfg.reconfig_us
             << " schedule=" << cfg.schedule_file
             << endl;

        RailOcsStaticSchedule sched;
        // ToR count for RailOCS depends on optional servers_per_tor splitting.
        uint32_t rail_splits = (rail_servers + (rail_servers_per_tor - 1)) / rail_servers_per_tor;
        uint32_t rail_tors = rail_gpus_per_server * rail_splits;
        if (!load_rail_ocs_static_schedule_slot0(cfg.schedule_file, rail_tors, rail_spines, sched)) {
            cout << "Failed to load RailOCS schedule file: " << cfg.schedule_file << endl;
            exit(1);
        }

        // Allow a separate queue size for OCS ToR<->ToR circuit links (in packets).
        // If not specified, reuse the global queue size.
        mem_b ocs_queuesize = queuesize;
        if (cfg.ocs_queue_pkts > 0) {
            ocs_queuesize = memFromPkt(cfg.ocs_queue_pkts);
        }

        top = new RailOcsTopology(
            rail_servers,
            rail_gpus_per_server,
            rail_servers_per_tor,
            rail_spines,
            sched,
            linkspeed,
            queuesize,
            ocs_queuesize,
            hop_latency,
            switch_latency,
            ocs_hop_latency,
            ocs_switch_latency,
            cfg.ocs_no_drop,
            cfg.ocs_queue_type,
            cfg.ocs_cut_through,
            qt,
            snd_type,
            &logfile,
            &eventlist
        );
    } else if (use_rail) {
        if (rail_servers == 0 || rail_spines == 0 || rail_gpus_per_server == 0) {
            cout << "Invalid -rail args. Expect -rail <servers> <spines> [gpus_per_server]" << endl;
            exit_error(argv[0]);
        }
        uint32_t rail_nodes = rail_servers * rail_gpus_per_server;
        if (no_of_nodes == (uint32_t)DEFAULT_NODES) {
            // User didn't override -nodes; auto-set for convenience.
            no_of_nodes = rail_nodes;
        }
        if (no_of_nodes != rail_nodes) {
            cout << "Error: -nodes (" << no_of_nodes << ") must equal servers*gpus_per_server ("
                 << rail_nodes << ") for rail topology." << endl;
            exit(1);
        }
        top = new RailTopology(
            rail_servers,
            rail_gpus_per_server,
            rail_spines,
            linkspeed,
            queuesize,
            hop_latency,
            switch_latency,
            qt,
            snd_type,
            rail_servers_per_tor,
            &logfile,
            &eventlist
        );
    } else if (topo_file) {
        top = FatTreeTopology::load(topo_file, qlf, eventlist, queuesize, qt, snd_type);
    } else {
        FatTreeTopology::set_tiers(tiers);
        top = new FatTreeTopology(no_of_nodes, linkspeed, queuesize, qlf, 
                                  &eventlist, NULL, qt, hop_latency,
                                  switch_latency,
                                  snd_type);
    }
        
#endif

#ifdef OV_FAT_TREE
    OversubscribedFatTreeTopology* top = new OversubscribedFatTreeTopology(lf, &eventlist,ff);
#endif

#ifdef MH_FAT_TREE
    MultihomedFatTreeTopology* top = new MultihomedFatTreeTopology(lf, &eventlist,ff);
#endif

#ifdef STAR
    StarTopology* top = new StarTopology(lf, &eventlist,ff);
#endif

#ifdef BCUBE
    BCubeTopology* top = new BCubeTopology(lf, &eventlist,ff);
    cout << "BCUBE " << K << endl;
#endif

#ifdef VL2
    VL2Topology* top = new VL2Topology(lf, &eventlist,ff);
#endif

    if (log_switches) {
        top->add_switch_loggers(logfile, timeFromUs(20.0));
    }

    vector<const Route*>*** net_paths;
    net_paths = new vector<const Route*>**[no_of_nodes];

    int **path_refcounts;
    path_refcounts = new int*[no_of_nodes];

    int* is_dest = new int[no_of_nodes];
    
    for (size_t s = 0; s < no_of_nodes; s++) {
        is_dest[s] = 0;
        net_paths[s] = new vector<const Route*>*[no_of_nodes];
        path_refcounts[s] = new int[no_of_nodes];
        for (size_t d = 0; d < no_of_nodes; d++) {
            net_paths[s][d] = NULL;
            path_refcounts[s][d] = 0;
        }
    }
    
    ConnectionMatrix* conns = new ConnectionMatrix(no_of_nodes);

    if (tm_file){
        cout << "Loading connection matrix from  " << tm_file << endl;

        if (!conns->load(tm_file)){
            cout << "Failed to load connection matrix " << tm_file << endl;
            exit(-1);
        }
    }
    else {
        cout << "Loading connection matrix from  standard input" << endl;        
        conns->load(cin);
    }

    if (conns->N != no_of_nodes){
        cout << "Connection matrix number of nodes is " << conns->N << " while I am using " << no_of_nodes << endl;
        exit(-1);
    }
    
    //handle link failures specified in the connection matrix.
    for (size_t c = 0; c < conns->failures.size(); c++){
        failure* crt = conns->failures.at(c);

        cout << "Adding link failure switch type" << crt->switch_type << " Switch ID " << crt->switch_id << " link ID "  << crt->link_id << endl;
        // Failures are currently implemented for FatTree switch models.
        auto* ft = dynamic_cast<FatTreeTopology*>(top);
        if (!ft) {
            cout << "Error: link failures are only supported for FatTree-based topologies." << endl;
            exit(1);
        }
        ft->add_failed_link(crt->switch_type,crt->switch_id,crt->link_id);
    }

    vector<NdpPullPacer*> pacers;

    for (size_t ix = 0; ix < no_of_nodes; ix++)
        pacers.push_back(new NdpPullPacer(eventlist,  linkspeed, 0.99));   

    // used just to print out stats data at the end
    list <const Route*> routes;

    vector<connection*>* all_conns = conns->getAllConnections();
    vector <NdpSrc*> ndp_srcs;

    for (size_t c = 0; c < all_conns->size(); c++){
        connection* crt = all_conns->at(c);
        int src = crt->src;
        int dest = crt->dst;
        path_refcounts[src][dest]++;
        path_refcounts[dest][src]++;
                        
        if (!net_paths[src][dest]
            && route_strategy!=ECMP_FIB
            && route_strategy!=ECMP_FIB_ECN
            && route_strategy!=REACTIVE_ECN ) {
            vector<const Route*>* paths = top->get_bidir_paths(src,dest,false);
            if (!paths || paths->empty()) {
                cout << "Error: no path between src=" << src << " and dest=" << dest
                     << " in the selected topology. "
                     << "For RailOCS (static mapping), this usually means your schedule does not connect the two ToRs.\n";
                exit(1);
            }
            net_paths[src][dest] = paths;
            /*
            for (unsigned int i = 0; i < paths->size(); i++) {
              routes.push_back((*paths)[i]);
            }
            */
        }
        if (!net_paths[dest][src]
            && route_strategy!=ECMP_FIB
            && route_strategy!=ECMP_FIB_ECN
            && route_strategy!=REACTIVE_ECN ) {
            vector<const Route*>* paths = top->get_bidir_paths(dest,src,false);
            if (!paths || paths->empty()) {
                cout << "Error: no path between src=" << dest << " and dest=" << src
                     << " in the selected topology. "
                     << "For RailOCS (static mapping), this usually means your schedule does not connect the two ToRs.\n";
                exit(1);
            }
            net_paths[dest][src] = paths;
            /*
            for (unsigned int i = 0; i < paths->size(); i++) {
              routes.push_back((*paths)[i]);
            }
            */
        }
    }

    map <flowid_t, TriggerTarget*> flowmap;

    for (size_t c = 0; c < all_conns->size(); c++){
        connection* crt = all_conns->at(c);
        int src = crt->src;
        int dest = crt->dst;
        //cout << "Connection " << crt->src << "->" <<crt->dst << " starting at " << crt->start << " size " << crt->size << endl;

        ndpSrc = new NdpSrc(NULL, NULL, eventlist,rts);
        ndpSrc->setCwnd(cwnd*Packet::data_packet_size());
        ndp_srcs.push_back(ndpSrc);
        ndpSrc->set_dst(dest);
        ndpSrc->set_path_burst(path_burst);
        if (crt->flowid) {
            ndpSrc->set_flowid(crt->flowid);
            assert(flowmap.find(crt->flowid) == flowmap.end()); // don't have dups
            flowmap[crt->flowid] = ndpSrc;
        }
                        
        if (crt->size>0){
            ndpSrc->set_flowsize(crt->size);
        }

        if (crt->trigger) {
            Trigger* trig = conns->getTrigger(crt->trigger, eventlist);
            trig->add_target(*ndpSrc);
        }
        if (crt->send_done_trigger) {
            Trigger* trig = conns->getTrigger(crt->send_done_trigger, eventlist);
            ndpSrc->set_end_trigger(*trig);
        }

        ndpSnk = new NdpSink(pacers[dest]);
                        
        ndpSrc->setName("ndp_" + ntoa(src) + "_" + ntoa(dest));

        //cout << "ndp_" + ntoa(src) + "_" + ntoa(dest) << endl;
        logfile.writeName(*ndpSrc);

        ndpSnk->set_src(src);
                        
        ndpSnk->setName("ndp_sink_" + ntoa(src) + "_" + ntoa(dest));
        logfile.writeName(*ndpSnk);
        if (crt->recv_done_trigger) {
            Trigger* trig = conns->getTrigger(crt->recv_done_trigger, eventlist);
            ndpSnk->set_end_trigger(*trig);
        }

        ndpSnk->set_priority(crt->priority);
                        
        ndpRtxScanner.registerNdp(*ndpSrc);

        switch (route_strategy) {
        case SCATTER_PERMUTE:
        case SCATTER_RANDOM:
        case SCATTER_ECMP:
        case PULL_BASED:
            ndpSrc->connect(NULL, NULL, *ndpSnk, crt->start);
            ndpSrc->set_paths(net_paths[src][dest]);
            ndpSnk->set_paths(net_paths[dest][src]);
            break;
        case ECMP_FIB:
        case ECMP_FIB_ECN:
        case REACTIVE_ECN:
            {
                auto* ft = dynamic_cast<FatTreeTopology*>(top);
                if (!ft) {
                    cout << "Error: ECMP_FIB/ECMP_FIB_ECN/REACTIVE_ECN require FatTreeTopology (switch model). "
                         << "Rail topology currently supports ecmp/perm/rand/pull style routing only." << endl;
                    exit(1);
                }
                Route* srctotor = new Route();
                srctotor->push_back(ft->queues_ns_nlp[src][ft->HOST_POD_SWITCH(src)][0]);
                srctotor->push_back(ft->pipes_ns_nlp[src][ft->HOST_POD_SWITCH(src)][0]);
                srctotor->push_back(ft->queues_ns_nlp[src][ft->HOST_POD_SWITCH(src)][0]->getRemoteEndpoint());

                Route* dsttotor = new Route();
                dsttotor->push_back(ft->queues_ns_nlp[dest][ft->HOST_POD_SWITCH(dest)][0]);
                dsttotor->push_back(ft->pipes_ns_nlp[dest][ft->HOST_POD_SWITCH(dest)][0]);
                dsttotor->push_back(ft->queues_ns_nlp[dest][ft->HOST_POD_SWITCH(dest)][0]->getRemoteEndpoint());


                ndpSrc->connect(srctotor, dsttotor, *ndpSnk, crt->start);
                ndpSrc->set_paths(path_entropy_size);
                ndpSnk->set_paths(path_entropy_size);

                //register src and snk to receive packets from their respective TORs. 
                assert(ft->switches_lp[ft->HOST_POD_SWITCH(src)]);
                assert(ft->switches_lp[ft->HOST_POD_SWITCH(src)]);
                ft->switches_lp[ft->HOST_POD_SWITCH(src)]->addHostPort(src,ndpSrc->flow_id(),ndpSrc);
                ft->switches_lp[ft->HOST_POD_SWITCH(dest)]->addHostPort(dest,ndpSrc->flow_id(),ndpSnk);
                break;
            }
        case SINGLE_PATH:
            {
                assert(route_strategy==SINGLE_PATH);
                int choice = rand()%net_paths[src][dest]->size();
                routeout = new Route(*(net_paths[src][dest]->at(choice)));
                routeout->add_endpoints(ndpSrc, ndpSnk);
                                
                routein = new Route(*top->get_bidir_paths(dest,src,false)->at(choice));
                routein->add_endpoints(ndpSnk, ndpSrc);
                ndpSrc->connect(routeout, routein, *ndpSnk, crt->start);
                break;
            }
        case NOT_SET:
            abort();
        }

        path_refcounts[src][dest]--;
        path_refcounts[dest][src]--;


        // set up the triggers
        // xxx

        // free up the routes if no other connection needs them 
        if (path_refcounts[src][dest] == 0 && net_paths[src][dest]) {
            vector<const Route*>::iterator i;
            for (i = net_paths[src][dest]->begin(); i != net_paths[src][dest]->end(); i++) {
                if ((*i)->reverse())
                    delete (*i)->reverse();
                delete *i;
            }
            delete net_paths[src][dest];
        }
        if (path_refcounts[dest][src] == 0 && net_paths[dest][src]) {
            vector<const Route*>::iterator i;
            for (i = net_paths[dest][src]->begin(); i != net_paths[dest][src]->end(); i++) {
                if ((*i)->reverse())
                    delete (*i)->reverse();
                delete *i;
            }
            delete net_paths[dest][src];
        }

        if (log_sink) {
            sinkLogger.monitorSink(ndpSnk);
        }
    }

    for (size_t ix = 0; ix < no_of_nodes; ix++) {
        delete path_refcounts[ix];
    }

    Logged::dump_idmap();
    // Record the setup
    int pktsize = Packet::data_packet_size();
    logfile.write("# pktsize=" + ntoa(pktsize) + " bytes");
    logfile.write("# hostnicrate = " + ntoa(linkspeed/1000000) + " Mbps");
    //logfile.write("# corelinkrate = " + ntoa(HOST_NIC*CORE_TO_HOST) + " pkt/sec");
    //logfile.write("# buffer = " + ntoa((double) (queues_na_ni[0][1]->_maxsize) / ((double) pktsize)) + " pkt");
    double rtt = timeAsSec(timeFromUs(RTT));
    logfile.write("# rtt =" + ntoa(rtt));
    
    // GO!
    cout << "Starting simulation" << endl;
    while (eventlist.doNextEvent()) {
    }

    cout << "Done" << endl;
    int new_pkts = 0, rtx_pkts = 0, bounce_pkts = 0;
    for (size_t ix = 0; ix < ndp_srcs.size(); ix++) {
        new_pkts += ndp_srcs[ix]->_new_packets_sent;
        rtx_pkts += ndp_srcs[ix]->_rtx_packets_sent;
        bounce_pkts += ndp_srcs[ix]->_bounces_received;
    }
    cout << "New: " << new_pkts << " Rtx: " << rtx_pkts << " Bounced: " << bounce_pkts << endl;
    /*
    list <const Route*>::iterator rt_i;
    int counts[10]; int hop;
    for (int i = 0; i < 10; i++)
        counts[i] = 0;
    cout << "route count: " << routes.size() << endl;
    for (rt_i = routes.begin(); rt_i != routes.end(); rt_i++) {
        const Route* r = (*rt_i);
        //print_route(*r);
#ifdef PRINTPATHS
        cout << "Path:" << endl;
#endif
        hop = 0;
        for (int i = 0; i < r->size(); i++) {
            PacketSink *ps = r->at(i); 
            CompositeQueue *q = dynamic_cast<CompositeQueue*>(ps);
            if (q == 0) {
#ifdef PRINTPATHS
                cout << ps->nodename() << endl;
#endif
            } else {
#ifdef PRINTPATHS
                cout << q->nodename() << " " << q->num_packets() << "pkts " 
                     << q->num_headers() << "hdrs " << q->num_acks() << "acks " << q->num_nacks() << "nacks " << q->num_stripped() << "stripped"
                     << endl;
#endif
                counts[hop] += q->num_stripped();
                hop++;
            }
        } 
#ifdef PRINTPATHS
        cout << endl;
#endif
    }
    for (int i = 0; i < 10; i++)
        cout << "Hop " << i << " Count " << counts[i] << endl;
    */  
}

