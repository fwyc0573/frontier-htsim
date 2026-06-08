// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_OCS_CONFIG_H
#define RAIL_OCS_CONFIG_H

#include "config.h"

#include <cstdint>
#include <istream>
#include <string>

// Minimal config for Rail+OCS topology, loaded from a .topo file.
struct RailOcsConfig {
    uint32_t servers = 0;
    uint32_t gpus_per_server = 8;
    uint32_t planes = 0;
    // Optional: split each rail (gpu_index) into multiple ToRs by grouping servers.
    // If servers_per_tor < servers, then each rail has ceil(servers/servers_per_tor) ToRs.
    // This mirrors RailTopology's servers_per_tor and enables inter-ToR traffic even for
    // fixed gpu_index collectives (e.g., CP all-to-all across servers).
    uint32_t servers_per_tor = 0; // 0 => no split (defaults to servers)

    // If set, used as the link speed for all rail/OCS links.
    double link_speed_gbps = 0.0;

    // Optional: queue sizing override for OCS ToR<->ToR links only.
    // Units match main_ndp's -q: number of packets (converted via memFromPkt()).
    // If 0, main_ndp will reuse the global queuesize for OCS links.
    //
    // Rationale: an OCS circuit fabric typically has negligible buffering; setting
    // OcsQueuePkts=1 approximates "no extra buffering beyond serialization".
    uint32_t ocs_queue_pkts = 0;
    // Optional: if set, OCS ToR<->ToR links will never drop due to buffer overflow.
    // This is useful to approximate an "ideal circuit" that does not lose packets.
    // Note: this can still create queueing delay under oversubscription, but avoids drops.
    bool ocs_no_drop = false;
    // Optional: choose the queue model used on OCS links when ocs_no_drop == false.
    // Supported values (case-insensitive):
    // - "" / "fifo": basic FIFO Queue (legacy behavior)
    // - "composite": CompositeQueue (supports trim/ECN-like behaviors, closer to fattree composite)
    // - "random": RandomQueue
    std::string ocs_queue_type = "";

    // Optional: cut-through modeling for OCS circuits.
    // When enabled, we model the OCS fabric as cut-through (no internal store-and-forward),
    // while preserving bandwidth constraints by keeping serialization/queueing at the
    // transmitter side (one egress serializer per (plane,src_tor), since schedule is a matching).
    //
    // Default: false (legacy behavior: each ToR<->ToR circuit edge has its own queue+pipe).
    bool ocs_cut_through = false;

    uint32_t link_latency_ns = 1000;
    uint32_t switch_latency_ns = 0;

    // Optional: latency override for OCS ToR<->ToR circuit links only.
    // If 0, main_ndp will reuse LinkLatencyNs/SwitchLatencyNs for OCS links.
    uint32_t ocs_link_latency_ns = 0;
    uint32_t ocs_switch_latency_ns = 0;

    double slot_us = 0.0;
    double reconfig_us = 0.0;

    std::string schedule_file;

    uint32_t nodes() const { return servers * gpus_per_server; }
    uint32_t rail_splits() const {
        uint32_t spt = servers_per_tor ? servers_per_tor : servers;
        return (servers + (spt - 1)) / spt;
    }
    uint32_t tors() const { return gpus_per_server * rail_splits(); }
};

bool load_rail_ocs_config(const char* filename, RailOcsConfig& out_cfg);
bool load_rail_ocs_config(std::istream& in, RailOcsConfig& out_cfg);

#endif


