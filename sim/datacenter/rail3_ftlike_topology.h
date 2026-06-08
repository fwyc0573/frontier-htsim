// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL3_FTLIKE_TOPOLOGY_H
#define RAIL3_FTLIKE_TOPOLOGY_H

#include "topology.h"
#include "eventlist.h"
#include "logfile.h"
#include "pipe.h"

#include "rail3_config.h"
#include "fat_tree_topology.h" // for queue_type and FEEDER_BUFFER behavior alignment

#include <cstdint>
#include <vector>

class RailDelaySwitch;

// Rail3FTLike: same structural fabric as Rail3 (ToR-Agg-Core), but with semantics aligned to
// current FatTreeTopology behavior under queue_type=composite for fair comparisons.
//
// Key differences vs Rail3Topology:
// - Does NOT explicitly insert per-switch "switch latency" Pipes into the route.
//   (FatTreeTopology in COMPOSITE mode typically does not push switch objects into the Route,
//    so switch pipeline delay may not be applied on the critical path.)
// - Host "src" queue uses FEEDER_BUFFER sizing similar to FatTreeTopology::alloc_src_queue.
//
// This avoids breaking the original Rail3Topology while allowing controlled A/B comparisons.
class Rail3FtLikeTopology final : public Topology {
public:
    Rail3FtLikeTopology(const Rail3Config& cfg,
                        linkspeed_bps linkspeed,
                        mem_b switch_queuesize_bytes,
                        simtime_picosec link_latency,
                        simtime_picosec switch_latency, // used only if cfg.use_switch_nodes=true
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
    uint32_t _agg_count;
    uint32_t _agg_up;
    uint32_t _core_count;
    uint32_t _no_of_nodes;

    // Link/queue params
    linkspeed_bps _linkspeed;
    mem_b _switch_queuesize_bytes;
    simtime_picosec _link_latency;
    simtime_picosec _switch_latency; // inserted as RailDelaySwitch pipeline delay if enabled
    queue_type _switch_queue_type;
    queue_type _host_queue_type;

    Logfile* _logfile;
    EventList* _eventlist;

    // Optional switch-node modeling (FatTree-like pipeline at ToR/Agg/Core).
    bool _use_switch_nodes = false;
    std::vector<RailDelaySwitch*> _tor_switches;  // [tor_global]
    std::vector<RailDelaySwitch*> _agg_switches;  // [pod*_agg_count + agg_in_pod]
    std::vector<RailDelaySwitch*> _core_switches; // [core_id]

    // Host <-> ToR (global host id / global ToR id).
    std::vector<Edge> _host_to_tor;              // [host]
    std::vector<std::vector<Edge>> _tor_to_host; // [tor][host]

    // ToR <-> Agg (within a pod).
    std::vector<std::vector<std::vector<Edge>>> _tor_to_agg; // [pod][tor_in_pod][agg_in_pod]
    std::vector<std::vector<std::vector<Edge>>> _agg_to_tor; // [pod][agg_in_pod][tor_in_pod]

    // Agg <-> Core.
    std::vector<std::vector<std::vector<Edge>>> _agg_to_core; // [pod][agg_in_pod][u]
    std::vector<std::vector<Edge>> _core_to_agg;               // [pod][core]

    uint32_t pod_of_host(uint32_t host) const { return host / _podsize; }
    uint32_t local_host(uint32_t host) const { return host % _podsize; }
    uint32_t rail_of_local_host(uint32_t h) const { return h % _gpus_per_server; }
    uint32_t server_of_local_host(uint32_t h) const { return h / _gpus_per_server; }
    uint32_t tor_in_pod_of_host(uint32_t host) const;
    uint32_t tor_global(uint32_t pod, uint32_t tor_in_pod) const { return pod * _tors_per_pod + tor_in_pod; }

    BaseQueue* make_switch_queue(const std::string& name);
    HostQueue* make_host_queue_ftlike(const std::string& name);
    Pipe* make_pipe(const std::string& name);

    void init_host_tor_edges();
    void init_fabric_edges();

    void init_switch_nodes();
    RailDelaySwitch* tor_switch(uint32_t tor_global_id) const;
    RailDelaySwitch* agg_switch(uint32_t pod, uint32_t agg_in_pod) const;
    RailDelaySwitch* core_switch(uint32_t core_id) const;
};

#endif // RAIL3_FTLIKE_TOPOLOGY_H


