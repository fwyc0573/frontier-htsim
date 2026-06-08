// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_TOPOLOGY_H
#define RAIL_TOPOLOGY_H

#include "topology.h"
#include "config.h"
#include "eventlist.h"
#include "logfile.h"
#include "loggers.h"
#include "randomqueue.h"
#include "pipe.h"
#include "fat_tree_topology.h"

#include <cstdint>
#include <vector>

// Rail-optimized topology:
// - Each "node" is a GPU endpoint.
// - GPUs are grouped by gpu_index into rails.
// - Each rail has a ToR; all GPUs with the same gpu_index connect to that ToR.
// - ToRs connect to all spines (full mesh). Different rails communicate via a spine.
//
// Node ID mapping:
//   node_id = server_id * gpus_per_server + gpu_index
//   gpu_index = node_id % gpus_per_server
//
// This models a common "rail" design where each server has multiple NICs / ports
// (one per rail), and traffic from a given GPU index tends to follow its rail.
class RailTopology final : public Topology {
public:
    RailTopology(uint32_t servers,
                 uint32_t gpus_per_server,
                 uint32_t spines,
                 linkspeed_bps linkspeed,
                 mem_b queuesize_bytes,
                 simtime_picosec link_latency,
                 simtime_picosec switch_latency,
                queue_type switch_queue_type,
                 queue_type host_queue_type,
                 uint32_t servers_per_tor,
                 Logfile* logfile,
                 EventList* eventlist);

    vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) override;
    vector<uint32_t>* get_neighbours(uint32_t src) override;
    uint32_t no_of_nodes() const override { return _no_of_nodes; }

    // Optional: record total switch queues at ToRs and spines.
    void add_switch_loggers(Logfile& log, simtime_picosec sample_period) override;

private:
    struct Edge {
        BaseQueue* q = nullptr;
        Pipe* p = nullptr;
    };

    uint32_t _servers;
    uint32_t _gpus_per_server;
    uint32_t _servers_per_tor; // max hosts per ToR per rail (0 => no split)
    uint32_t _rail_splits;     // number of ToRs per rail
    uint32_t _tors;   // = gpus_per_server * _rail_splits
    uint32_t _spines;
    uint32_t _no_of_nodes;

    linkspeed_bps _linkspeed;
    mem_b _queuesize_bytes;
    simtime_picosec _link_latency;
    simtime_picosec _switch_latency;
    queue_type _switch_queue_type;
    queue_type _host_queue_type;

    Logfile* _logfile;
    EventList* _eventlist;

    // Host <-> ToR edges
    // host_to_tor[h] uses ToR rail(h); tor_to_host[rail][h]
    std::vector<Edge> _host_to_tor;
    std::vector<std::vector<Edge>> _tor_to_host;

    // ToR <-> Spine edges (full mesh)
    std::vector<std::vector<Edge>> _tor_to_spine;   // [tor][spine]
    std::vector<std::vector<Edge>> _spine_to_tor;   // [spine][tor]

    uint32_t rail_of_host(uint32_t host) const { return host % _gpus_per_server; }
    uint32_t server_of_host(uint32_t host) const { return host / _gpus_per_server; }
    uint32_t tor_of_host(uint32_t host) const;

    void init_network();
    BaseQueue* make_switch_queue(const std::string& name);
    HostQueue* make_host_queue(const std::string& name);
    Pipe* make_pipe(const std::string& name);
};

#endif


