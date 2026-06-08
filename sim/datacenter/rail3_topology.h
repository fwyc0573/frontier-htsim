// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL3_TOPOLOGY_H
#define RAIL3_TOPOLOGY_H

#include "topology.h"
#include "config.h"
#include "eventlist.h"
#include "logfile.h"
#include "loggers.h"
#include "pipe.h"

#include "rail3_config.h"
#include "fat_tree_topology.h" // for queue_type enum (legacy shared typedef)

#include <cstdint>
#include <vector>

// Rail3: 3-tier packet-switched ToR-Agg-Core fabric while preserving the
// Rail host->ToR mapping (group by rail index, optional ToR split via ServersPerTor).
//
// Compared with FatTreeTopology:
// - Host numbering / ToR assignment matches Rail / RailOCS3 (not fattree HOST_POD_SWITCH()).
// - Top-of-fabric is a regular core set; ECMP is modeled by enumerating cores.
//
// Connectivity model (default NB-ish parameters):
// - tors_per_pod = gpus_per_server * rail_splits_per_pod
// - AggCount defaults to TorUp; each ToR connects to all Aggs with 1 link.
// - Each Agg connects down to all ToRs in its pod.
// - Each Agg has AggUp uplinks to Core; CoreCount = AggCount * AggUp.
// - Each Core switch connects to one Agg per pod (same agg index = core/AggUp).
class Rail3Topology final : public Topology {
public:
    Rail3Topology(const Rail3Config& cfg,
                  linkspeed_bps linkspeed,
                  mem_b switch_queuesize_bytes,
                  simtime_picosec link_latency,
                  simtime_picosec switch_latency,
                  queue_type switch_queue_type,
                  queue_type host_queue_type,
                  Logfile* logfile,
                  EventList* eventlist);

    vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) override;
    vector<uint32_t>* get_neighbours(uint32_t src) override;
    uint32_t no_of_nodes() const override { return _no_of_nodes; }

private:
    struct Edge {
        BaseQueue* q = nullptr;
        Pipe* p = nullptr;
    };

    // Structure params
    uint32_t _pods;
    uint32_t _podsize;
    uint32_t _gpus_per_server;
    uint32_t _servers_per_tor;
    uint32_t _servers_per_pod;
    uint32_t _rail_splits_per_pod;
    uint32_t _tors_per_pod;
    uint32_t _tor_up;
    uint32_t _agg_count; // per pod
    uint32_t _agg_up;
    uint32_t _core_count;
    uint32_t _no_of_nodes;

    // Link/queue params
    linkspeed_bps _linkspeed;
    mem_b _switch_queuesize_bytes;
    simtime_picosec _link_latency;
    simtime_picosec _switch_latency;
    queue_type _switch_queue_type;
    queue_type _host_queue_type;

    Logfile* _logfile;
    EventList* _eventlist;

    // Host <-> ToR (global host id / global ToR id).
    std::vector<Edge> _host_to_tor;              // [host]
    std::vector<std::vector<Edge>> _tor_to_host; // [tor][host]

    // ToR <-> Agg (within a pod).
    std::vector<std::vector<std::vector<Edge>>> _tor_to_agg; // [pod][tor_in_pod][agg_in_pod]
    std::vector<std::vector<std::vector<Edge>>> _agg_to_tor; // [pod][agg_in_pod][tor_in_pod]

    // Agg <-> Core.
    // For each pod, agg_in_pod and uplink u map to core = agg_in_pod * _agg_up + u.
    std::vector<std::vector<std::vector<Edge>>> _agg_to_core; // [pod][agg_in_pod][u]
    std::vector<std::vector<Edge>> _core_to_agg;               // [pod][core]

    uint32_t pod_of_host(uint32_t host) const { return host / _podsize; }
    uint32_t local_host(uint32_t host) const { return host % _podsize; }
    uint32_t rail_of_local_host(uint32_t h) const { return h % _gpus_per_server; }
    uint32_t server_of_local_host(uint32_t h) const { return h / _gpus_per_server; }
    uint32_t tor_in_pod_of_host(uint32_t host) const;
    uint32_t tor_global(uint32_t pod, uint32_t tor_in_pod) const { return pod * _tors_per_pod + tor_in_pod; }

    BaseQueue* make_switch_queue(const std::string& name);
    HostQueue* make_host_queue(const std::string& name);
    Pipe* make_pipe(const std::string& name);

    void init_host_tor_edges();
    void init_fabric_edges();
};

#endif // RAIL3_TOPOLOGY_H


