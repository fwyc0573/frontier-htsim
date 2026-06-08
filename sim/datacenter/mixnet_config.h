// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef MIXNET_CONFIG_H
#define MIXNET_CONFIG_H

#include <cstdint>
#include <istream>
#include <string>
#include <vector>

// Minimal config for MixNet-like "dual fabric" topology (EPS + OCS), loaded from a .topo file.
//
// NOTE(scope): Current implementation covers:
// - EPS is modeled as a packet-switched fat-tree loaded from a standard FatTreeTopology .topo file.
// - OCS is modeled as either RailOCS or RailOCS3 topo file (existing formats).
// - Routing policy is group-based: for (src,dst) in the same EP-group => use OCS paths; else use EPS paths.
//
// Note: NIC split (2 NICs to EPS, 6 NICs to OCS) and intra-server gateway forwarding are NOT modeled yet.
// This config is intentionally focused on validating EP-group-limited OCS usage first.
struct MixNetConfig {
    // Physical node_id mapping is assumed to follow Rail-style host mapping:
    // node_id = server_id * gpus_per_server + local_gpu_index
    // This is used to model "gateway NICs" (subset of GPUs can inject to EPS).
    uint32_t gpus_per_server = 8;
    // Local GPU indices (within a server) that have EPS-connected NICs (e.g., 0 1).
    // These are the only nodes that can directly inject to EPS; others must forward to a gateway.
    std::vector<uint32_t> eps_gateway_local_indices;

    std::string eps_topo_file;
    // Group-local OCS (static slot 0) modeled as a single OCS crossbar per EP group.
    // Schedule format matches RailOCS static schedule: "slot 0" + "plane p a-b c-d ..."
    // with endpoints a,b,... being local port indices within an EP group.
    //
    // Note: the schedule is applied identically to every EP group.
    std::string ocs_schedule_file;
    uint32_t ocs_planes = 0;
    // OCS link speed for the direct circuit edges (Gbps). If 0, main_ndp -linkspeed is used.
    double ocs_link_speed_gbps = 0.0;
    // Queue sizing override for OCS direct circuit edges only (packets). If 0, reuse global -q.
    uint32_t ocs_queue_pkts = 0;
    bool ocs_no_drop = false;
    // Optional latency override for OCS circuit edges only. If 0, reuse global hop/switch latency.
    uint32_t ocs_link_latency_ns = 0;
    uint32_t ocs_switch_latency_ns = 0;
    std::string ep_group_map_file;
};

bool load_mixnet_config(const char* filename, MixNetConfig& out_cfg);
bool load_mixnet_config(std::istream& in, MixNetConfig& out_cfg);

#endif // MIXNET_CONFIG_H

