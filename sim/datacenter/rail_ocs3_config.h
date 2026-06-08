// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_OCS3_CONFIG_H
#define RAIL_OCS3_CONFIG_H

#include "config.h"

#include <cstdint>
#include <istream>
#include <string>

// Minimal config for Rail-OCS3 (3-layer, 2-OCS, B1) topology, loaded from a .topo file.
//
// This config is intentionally "schedule-centric":
// - Agg-OCS schedule operates on port-level endpoints (ToR uplinks + trunk ports), per pod.
// - Core-OCS schedule operates on port-level endpoints (trunk ports across pods).
struct RailOcs3Config {
    // Number of pods.
    uint32_t pods = 0;
    // Hosts (GPU ranks) per pod.
    uint32_t podsize = 0;

    uint32_t gpus_per_server = 8;
    // Optional: split each rail (gpu_index) into multiple ToRs by grouping servers.
    // If servers_per_tor < servers_per_pod, then each rail has ceil(servers_per_pod/servers_per_tor) ToRs.
    uint32_t servers_per_tor = 0; // 0 => defaults to servers_per_pod (no split)

    // Per-ToR uplink ports into Agg-OCS (port-level endpoints).
    uint32_t tor_up = 0;
    // Number of trunk ports per pod between Agg-OCS and Core-OCS.
    uint32_t trunk_ports_per_pod = 0;

    // Schedule parameters (slot0-only schedules).
    uint32_t agg_ocs_planes = 0;
    uint32_t core_ocs_planes = 0;
    std::string agg_ocs_schedule_file;
    std::string core_ocs_schedule_file;

    // Link speed for all links (Gbps).
    double link_speed_gbps = 0.0;

    // Queue sizing override for OCS links only (in packets, like main_ndp -q).
    uint32_t ocs_queue_pkts = 0;
    bool ocs_no_drop = false;
    // Optional: choose the queue model used on OCS links when ocs_no_drop == false.
    // Supported values (case-insensitive):
    // - "" / "fifo": basic FIFO Queue (legacy behavior)
    // - "composite": CompositeQueue (supports trim/ECN-like behaviors, matching fattree composite)
    // - "random": RandomQueue
    std::string ocs_queue_type = "";

    // New: if true, model OCS circuits as cut-through (no internal store-and-forward queueing
    // inside the OCS fabric), while still preserving link bandwidth constraints by keeping
    // ToR egress serialization/queueing on ToR uplinks.
    //
    // Concretely when enabled:
    // - ToR->(OCS) edges keep an OCS queue (rate-limited to linkspeed) + OCS pipe (latency)
    // - All internal trunk/core edges become Pipe-only (no additional queues/serialization)
    //
    // This is closer to a static optical circuit behaving like a dedicated wire:
    // contention/queueing happens at the transmitter (ToR uplink), not at intermediate OCS ports.
    //
    // Default: false (legacy behavior: OCS segments each have their own queue + pipe).
    bool ocs_cut_through = false;

    // If true, size host source queues like FatTreeTopology::alloc_src_queue:
    // memFromPkt(FEEDER_BUFFER) instead of using switch_queuesize_bytes.
    // This helps align sender-side buffering semantics across topologies.
    //
    // Default: false (legacy behavior).
    bool host_feeder_buffer = false;

    // Non-OCS links (host<->ToR) latency / switch latency.
    uint32_t link_latency_ns = 1000;
    uint32_t switch_latency_ns = 0;

    // Optional: latency override for OCS circuit links only.
    uint32_t ocs_link_latency_ns = 0;
    uint32_t ocs_switch_latency_ns = 0;

    double slot_us = 0.0;
    double reconfig_us = 0.0;

    // Derived helpers.
    uint32_t servers_per_pod() const { return podsize / gpus_per_server; }
    uint32_t servers() const { return pods * servers_per_pod(); }
    uint32_t nodes() const { return pods * podsize; }

    uint32_t rail_splits_per_pod() const {
        uint32_t spt = servers_per_tor ? servers_per_tor : servers_per_pod();
        return (servers_per_pod() + (spt - 1)) / spt;
    }
    uint32_t tors_per_pod() const { return gpus_per_server * rail_splits_per_pod(); }

    // Agg-OCS per-pod port count (port-level endpoints).
    uint32_t agg_ocs_ports_per_pod() const { return tors_per_pod() * tor_up + trunk_ports_per_pod; }
    // Core-OCS global port count (port-level endpoints).
    uint32_t core_ocs_ports_total() const { return pods * trunk_ports_per_pod; }
};

bool load_rail_ocs3_config(const char* filename, RailOcs3Config& out_cfg);
bool load_rail_ocs3_config(std::istream& in, RailOcs3Config& out_cfg);

#endif // RAIL_OCS3_CONFIG_H


