// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_OCS_TOPOLOGY_H
#define RAIL_OCS_TOPOLOGY_H

#include "topology.h"
#include "config.h"
#include "eventlist.h"
#include "logfile.h"
#include "pipe.h"
#include "queue.h"
#include "fat_tree_topology.h" // for queue_type (PRIORITY/FAIR_PRIO)

#include "rail_ocs_schedule.h"

#include <cstdint>
#include <string>
#include <vector>

// Rail + static OCS topology:
// - Each GPU is a host node.
// - Each rail (gpu_index) has a ToR.
// - Inter-ToR connectivity is provided by OCS planes with a fixed matching (slot 0).
//
// In this "no reconfig" assumption, connectivity never changes during the run.
class RailOcsTopology final : public Topology {
public:
    RailOcsTopology(uint32_t servers,
                    uint32_t gpus_per_server,
                    uint32_t servers_per_tor,
                    uint32_t planes,
                    const RailOcsStaticSchedule& schedule,
                    linkspeed_bps linkspeed,
                    mem_b switch_queuesize_bytes,
                    mem_b ocs_queuesize_bytes,
                    simtime_picosec link_latency,
                    simtime_picosec switch_latency,
                    simtime_picosec ocs_link_latency,
                    simtime_picosec ocs_switch_latency,
                    bool ocs_no_drop,
                    const std::string& ocs_queue_type,
                    bool ocs_cut_through,
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

    uint32_t _servers;
    uint32_t _gpus_per_server;
    uint32_t _servers_per_tor; // per rail
    uint32_t _rail_splits;     // number of ToRs per rail
    uint32_t _tors;
    uint32_t _planes;
    uint32_t _no_of_nodes;

    RailOcsStaticSchedule _schedule;

    linkspeed_bps _linkspeed;
    mem_b _switch_queuesize_bytes;
    mem_b _ocs_queuesize_bytes;
    simtime_picosec _link_latency;
    simtime_picosec _switch_latency;
    simtime_picosec _ocs_link_latency;
    simtime_picosec _ocs_switch_latency;
    bool _ocs_no_drop;
    std::string _ocs_queue_type;
    bool _ocs_cut_through;
    queue_type _switch_queue_type;
    queue_type _host_queue_type;

    Logfile* _logfile;
    EventList* _eventlist;

    // host -> ToR (host egress must be HostQueue)
    std::vector<Edge> _host_to_tor;
    // ToR -> host
    std::vector<std::vector<Edge>> _tor_to_host;

    // OCS plane links between ToRs: tor_to_tor[plane][src_tor][dst_tor]
    std::vector<std::vector<std::vector<Edge>>> _tor_to_tor;
    // Cut-through mode: per-(plane,src_tor) egress serializer queue (schedule is a matching).
    std::vector<std::vector<Edge>> _tor_ocs_egress; // [plane][src_tor]

    uint32_t rail_of_host(uint32_t host) const { return host % _gpus_per_server; }
    uint32_t server_of_host(uint32_t host) const { return host / _gpus_per_server; }
    uint32_t tor_of_host(uint32_t host) const;

    HostQueue* make_host_queue(const std::string& name);
    BaseQueue* make_switch_queue(const std::string& name);
    BaseQueue* make_ocs_queue(const std::string& name);
    BaseQueue* maybe_make_ocs_queue(const std::string& name, bool tor_egress);
    Pipe* make_pipe(const std::string& name);
    Pipe* make_ocs_pipe(const std::string& name);

    void init_network();
};

#endif


