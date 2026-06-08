// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_OCS3_TOPOLOGY_H
#define RAIL_OCS3_TOPOLOGY_H

#include "topology.h"
#include "config.h"
#include "eventlist.h"
#include "logfile.h"
#include "pipe.h"
#include "queue.h"
#include "fat_tree_topology.h" // for queue_type (PRIORITY/FAIR_PRIO)

#include "rail_ocs3_config.h"
#include "rail_ocs3_port_schedule.h"

#include <cstdint>
#include <string>
#include <vector>

// Rail-OCS3 topology (3-layer, 2-OCS, B1):
// - Hosts are GPUs (nodes).
// - ToR mapping follows Rail: per pod, one rail per GPU-index, with optional servers_per_tor splitting.
// - Agg-OCS (per pod) switches between ToR uplink ports and trunk ports (to Core-OCS), using a static port-level schedule (slot 0).
// - Core-OCS switches between trunk ports across pods, using a static port-level schedule (slot 0).
//
// =========================
// Current support & limits
// =========================
// NOTE(scope): This implementation is intentionally scoped to validate:
// - config parsing (RailOCS3 .topo)
// - port-level schedule parsing (slot0/plane matching)
// - basic end-to-end reachability
//
// Supported (implemented today):
// - Cross-pod routing (when schedules provide it):
//     ToR(src) -> Agg-OCS(ToR->trunk) -> Core-OCS(trunk->trunk) -> Agg-OCS(trunk->ToR) -> ToR(dst)
// - Intra-pod direct ToR<->ToR circuits if Agg-OCS schedule contains ToR-port <-> ToR-port pairs.
//
// LIMITATION: Known constraints in current routing implementation:
// - Port-level schedules are parsed, but then compressed to ToR-level peers per plane:
//   - For a given (pod, plane), each ToR is allowed at most ONE trunk peer (ToR->trunk) and
//     at most ONE ToR peer (ToR->ToR). If multiple uplinks of the same ToR appear in the same
//     plane (which is legal in port-level schedules), we currently reject (assert) to keep
//     route enumeration simple.
//   - As a result, for a given (pod, plane), a ToR is effectively "either connected to one trunk
//     or directly to one other ToR" (or not connected), from this topology's perspective.
// - Intra-pod multi-hop paths (e.g., ToR->trunk->Core-OCS->trunk->ToR within the same pod) are
//   NOT implemented; for src_pod==dst_pod we only return direct ToR<->ToR circuits.
// - Cross-pod routing currently requires that:
//   - Agg-OCS provides ToR->trunk on the src side AND trunk->ToR on the dst side
//   - Core-OCS provides trunk->trunk mapping that lands in dst pod
// - General "arbitrary topology connectivity given a schedule" is NOT the goal of this implementation.
//   If you need a fully general port-level router (multiple peers per ToR per plane, or arbitrary
//   endpoint classes), route enumeration must be upgraded to operate on (ToR, uplink_port) endpoints.
//
// Queue modeling note:
// - Non-OCS switch queues follow the global -queue_type (e.g., composite).
// - OCS link queues historically used simple FIFO Queue (or NoDropQueue). This can materially
//   change congestion/feedback behavior vs. fattree composite queues. To support apples-to-apples
//   comparisons, RailOCS3 .topo supports:
//     OcsQueueType fifo|composite|random
//   which is used when OcsNoDrop=0.
class RailOcs3Topology final : public Topology {
public:
    RailOcs3Topology(const RailOcs3Config& cfg,
                     const RailOcs3PortSchedule& agg_ocs_sched,
                     const RailOcs3PortSchedule& core_ocs_sched,
                     linkspeed_bps linkspeed,
                     mem_b switch_queuesize_bytes,
                     mem_b ocs_queuesize_bytes,
                     simtime_picosec link_latency,
                     simtime_picosec switch_latency,
                     simtime_picosec ocs_link_latency,
                     simtime_picosec ocs_switch_latency,
                     bool ocs_no_drop,
                     queue_type switch_queue_type,
                     queue_type host_queue_type,
                     Logfile* logfile,
                     EventList* eventlist);

    vector<const Route*>* get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) override;
    vector<uint32_t>* get_neighbours(uint32_t src) override;
    uint32_t no_of_nodes() const override { return _no_of_nodes; }

    // Debug helper: dump reachability information derived from the current routing logic
    // (i.e., exactly what get_bidir_paths() would return), to help debug schedules/configs.
    //
    // Output includes:
    // - ToR-level reachability matrix (using representative hosts per ToR)
    // - Summary stats + a bounded list of unreachable examples with coarse reasons
    //
    // This is intentionally not performance-critical; it is meant for debugging.
    void dump_reachability(const std::string& filename,
                           uint32_t max_unreachable_examples = 200,
                           uint32_t max_tor_pairs_print = 512);

private:
    struct Edge {
        BaseQueue* q = nullptr;
        Pipe* p = nullptr;
    };

    RailOcs3Config _cfg;
    uint32_t _no_of_nodes;
    uint32_t _pods;
    uint32_t _podsize;
    uint32_t _gpus_per_server;
    uint32_t _servers_per_tor;
    uint32_t _rail_splits_per_pod;
    uint32_t _tors_per_pod;
    uint32_t _tors_total;

    uint32_t _tor_up;
    uint32_t _trunk_ports_per_pod;
    uint32_t _agg_ocs_ports_per_pod;
    uint32_t _agg_ocs_planes;
    uint32_t _core_ocs_planes;
    uint32_t _core_ocs_ports_total;

    RailOcs3PortSchedule _agg_sched;
    RailOcs3PortSchedule _core_sched;

    linkspeed_bps _linkspeed;
    mem_b _switch_queuesize_bytes;
    mem_b _ocs_queuesize_bytes;
    simtime_picosec _link_latency;
    simtime_picosec _switch_latency;
    simtime_picosec _ocs_link_latency;
    simtime_picosec _ocs_switch_latency;
    bool _ocs_no_drop;
    bool _ocs_cut_through;
    bool _host_feeder_buffer;
    queue_type _switch_queue_type;
    queue_type _host_queue_type;
    Logfile* _logfile;
    EventList* _eventlist;

    // Host <-> ToR (global host id / global ToR id).
    std::vector<Edge> _host_to_tor;                 // [host]
    std::vector<std::vector<Edge>> _tor_to_host;    // [tor][host]

    // Agg-OCS edges (LIMITATION: port-level schedules are compressed to ToR-level peers per plane):
    // - Schedules are port-level, but we compress them to ToR-level peers per plane.
    // - Per pod, per plane:
    //   - each ToR has at most one trunk peer (ToR->trunk), and
    //   - each ToR has at most one ToR peer (ToR->ToR).
    // - trunk index is local within a pod: [0, trunk_ports_per_pod).
    std::vector<std::vector<std::vector<int32_t>>> _tor_to_trunk;   // [pod][plane][tor_in_pod] -> trunk or -1
    std::vector<std::vector<std::vector<int32_t>>> _trunk_to_tor;   // [pod][plane][trunk] -> tor_in_pod or -1
    std::vector<std::vector<std::vector<Edge>>> _agg_tor_to_trunk;  // [pod][plane][tor_in_pod]
    std::vector<std::vector<std::vector<Edge>>> _agg_trunk_to_tor;  // [pod][plane][trunk]

    // Optional pod-local direct ToR<->ToR circuits provided by Agg-OCS schedule.
    // LIMITATION: at most one ToR-to-ToR peer per ToR per plane.
    std::vector<std::vector<std::vector<int32_t>>> _tor_to_tor;     // [pod][plane][tor_in_pod] -> peer_tor_in_pod or -1
    std::vector<std::vector<std::vector<Edge>>> _agg_tor_to_tor;    // [pod][plane][tor_in_pod] (directed edge to peer)

    // Core-OCS edges (global trunk endpoint ids).
    std::vector<std::vector<Edge>> _core_link; // [plane][global_trunk_port]

    uint32_t pod_of_host(uint32_t host) const { return host / _podsize; }
    uint32_t local_host(uint32_t host) const { return host % _podsize; }

    uint32_t rail_of_local_host(uint32_t h) const { return h % _gpus_per_server; }
    uint32_t server_of_local_host(uint32_t h) const { return h / _gpus_per_server; }
    uint32_t tor_in_pod_of_host(uint32_t host) const;
    uint32_t tor_global(uint32_t pod, uint32_t tor_in_pod) const { return pod * _tors_per_pod + tor_in_pod; }

    HostQueue* make_host_queue(const std::string& name);
    BaseQueue* make_switch_queue(const std::string& name);
    BaseQueue* make_ocs_queue(const std::string& name);
    // Cut-through OCS mode:
    // - keep ToR egress serialization queues (tor_egress=true)
    // - remove internal OCS/trunk queues (tor_egress=false => nullptr)
    BaseQueue* maybe_make_ocs_queue(const std::string& name, bool tor_egress);
    Pipe* make_pipe(const std::string& name);
    Pipe* make_ocs_pipe(const std::string& name);

    void init_host_tor_edges();
    void init_agg_ocs_edges();
    void init_core_ocs_edges();

    // Debug helpers.
    bool has_intra_pod_direct_tor_circuit(uint32_t pod, uint32_t tor_a_in_pod, uint32_t tor_b_in_pod, uint32_t* out_plane = nullptr) const;
    bool has_src_tor_to_any_trunk(uint32_t pod, uint32_t tor_in_pod, uint32_t* out_plane = nullptr, uint32_t* out_trunk = nullptr) const;
    bool has_dst_trunk_to_tor(uint32_t pod, uint32_t trunk, uint32_t tor_in_pod, uint32_t* out_plane = nullptr) const;
    bool has_core_trunk_to_pod(uint32_t src_pod, uint32_t src_trunk, uint32_t dst_pod, uint32_t* out_plane = nullptr, uint32_t* out_dst_trunk = nullptr) const;
};

#endif // RAIL_OCS3_TOPOLOGY_H


