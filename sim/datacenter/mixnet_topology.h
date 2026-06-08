// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MIXNET_TOPOLOGY_H
#define MIXNET_TOPOLOGY_H

#include "topology.h"
#include "fat_tree_topology.h" // for queue_type / host queue types

#include <cstdint>
#include <string>
#include <vector>

// MixNetTopology: a "meta-topology" that delegates routing to one of two sub-topologies:
// - EPS fabric (typically fat-tree)
// - OCS fabric (typically RailOCS / RailOCS3)
//
// Routing policy:
// - If src and dst are in the same EP-group => return OCS paths
// - Else => route via per-server EPS gateways:
//     src -> gw(src_server) -> EPS -> gw(dst_server) -> dst
//
// Note: this class does not own eps/ocs pointers (they are expected to outlive MixNetTopology).
class MixNetTopology final : public Topology {
public:
    MixNetTopology(Topology* eps,
                   Topology* ocs,
                   std::vector<uint32_t> ep_group_id,
                   uint32_t gpus_per_server,
                   std::vector<uint32_t> eps_gateway_local_indices,
                   // "intra-server forwarding" used only for gateway reachability (cost can be made ~0).
                   linkspeed_bps intra_linkspeed,
                   mem_b intra_queue_bytes,
                   simtime_picosec intra_latency,
                   queue_type host_queue_type,
                   Logfile* logfile,
                   EventList* eventlist);

    vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) override;
    vector<uint32_t>* get_neighbours(uint32_t src) override;
    uint32_t no_of_nodes() const override { return _no_of_nodes; }

private:
    struct Edge {
        HostQueue* q = nullptr;
        Pipe* p = nullptr;
    };

    Topology* _eps;
    Topology* _ocs;
    std::vector<uint32_t> _ep_group_id; // [node] -> group id
    uint32_t _no_of_nodes;

    uint32_t _gpus_per_server;
    std::vector<uint32_t> _eps_gateway_local_indices;
    linkspeed_bps _intra_linkspeed;
    mem_b _intra_queue_bytes;
    simtime_picosec _intra_latency;
    queue_type _host_queue_type;
    Logfile* _logfile;
    EventList* _eventlist;

    // For each node and each gateway option k, a directed edge:
    // - to_gw[node][k]: node -> gw(server(node), gateway_local_indices[k])
    // - from_gw[node][k]: gw(server(node), gateway_local_indices[k]) -> node
    std::vector<std::vector<Edge>> _to_gw;   // [node][k]
    std::vector<std::vector<Edge>> _from_gw; // [node][k]
    // full node_id -> gw-only EPS node_id for gateway nodes; -1 for non-gateway nodes.
    std::vector<int32_t> _full_to_eps;

    uint32_t server_of(uint32_t node) const { return node / _gpus_per_server; }
    uint32_t local_gpu(uint32_t node) const { return node % _gpus_per_server; }
    uint32_t gw_node(uint32_t node, uint32_t k) const;
    int32_t gw_eps_id(uint32_t full_gw_node) const;

    HostQueue* make_intra_host_queue(const std::string& name);
    Pipe* make_intra_pipe(const std::string& name);
};

#endif // MIXNET_TOPOLOGY_H

