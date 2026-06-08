// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef EP_GROUP_OCS_TOPOLOGY_H
#define EP_GROUP_OCS_TOPOLOGY_H

#include "topology.h"
#include "rail_ocs_schedule.h"
#include "fat_tree_topology.h" // for queue_type enum

#include <cstdint>
#include <vector>

// EpGroupOcsTopology:
// - Each EP group has a dedicated OCS crossbar (single "switch"), but we model it as direct circuits.
// - Each node attaches to exactly one EP group and has a local port index (typically ep_idx).
// - A static schedule (slot 0) provides, per plane, a matching between local ports within a group.
// - Two nodes are connected in plane p iff schedule.peer[p][local_src] == local_dst.
//
// Routing semantics:
// - If src/dst are in different EP groups => no paths (unreachable).
// - If same group => return one path per plane that connects them.
class EpGroupOcsTopology final : public Topology {
public:
    EpGroupOcsTopology(const std::vector<uint32_t>& group_id,
                      const std::vector<uint32_t>& local_port,
                      uint32_t group_size,
                      const RailOcsStaticSchedule& schedule,
                      linkspeed_bps linkspeed,
                      mem_b host_queuesize_bytes,
                      mem_b ocs_queuesize_bytes,
                      simtime_picosec link_latency,
                      simtime_picosec switch_latency,
                      simtime_picosec ocs_link_latency,
                      simtime_picosec ocs_switch_latency,
                      bool ocs_no_drop,
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

    uint32_t _no_of_nodes;
    uint32_t _group_size;
    uint32_t _planes;
    std::vector<uint32_t> _group_id;
    std::vector<uint32_t> _local_port;
    RailOcsStaticSchedule _schedule;

    linkspeed_bps _linkspeed;
    mem_b _host_queuesize_bytes;
    mem_b _ocs_queuesize_bytes;
    simtime_picosec _link_latency;
    simtime_picosec _switch_latency;
    simtime_picosec _ocs_link_latency;
    simtime_picosec _ocs_switch_latency;
    bool _ocs_no_drop;
    queue_type _host_queue_type;
    Logfile* _logfile;
    EventList* _eventlist;

    std::vector<HostQueue*> _host_q; // [node]
    std::vector<Edge> _host_in;      // [node] receiver NIC queue+pipe
    std::vector<std::vector<Edge>> _ocs_out; // [plane][node] outgoing circuit to its paired peer (if any)

    HostQueue* make_host_queue(const std::string& name);
    BaseQueue* make_ocs_queue(const std::string& name);
    BaseQueue* make_host_in_queue(const std::string& name);
    Pipe* make_pipe(const std::string& name, simtime_picosec lat);
};

#endif // EP_GROUP_OCS_TOPOLOGY_H

