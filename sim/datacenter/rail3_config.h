// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL3_CONFIG_H
#define RAIL3_CONFIG_H

#include <cstdint>
#include <istream>
#include <string>

// A 3-tier packet-switched topology that reuses the Rail host->ToR mapping
// (GPUs grouped by rail index, optional ToR splitting via ServersPerTor),
// but replaces the rail "spine full-mesh" with a classic ToR-Agg-Core fabric.
//
// This is intended as a baseline for comparing with RailOCS3 (two OCS layers).
//
// Minimal .topo format:
//   Type Rail3
//   Pods <uint>
//   Podsize <uint>              # GPUs per pod
//   GpusPerServer <uint>
//   ServersPerTor <uint>        # per rail (0 => default=servers_per_pod)
//   TorUp <uint>                # uplinks per ToR (to Agg)
//   AggCount <uint>             # optional (0 => default=TorUp)
//   AggUp <uint>                # optional (0 => default=tors_per_pod)
//   LinkSpeedGbps <double>      # optional (0 => use CLI -linkspeed)
//   LinkLatencyNs <uint>        # optional
//   SwitchLatencyNs <uint>      # optional
struct Rail3Config {
    enum Mode {
        RAIL3 = 0,        // Original Rail3 implementation (explicit switch-latency Pipes in routes)
        RAIL3_FTLIKE = 1, // "FatTree3-like" semantics for fair comparison (see below)
    };
    Mode mode = RAIL3;

    // If true, insert explicit Switch nodes (with switch_latency) into routes, similar to FatTreeSwitch's
    // ingress->fabric->egress pipeline modeling. This is mainly useful for "apples-to-apples" semantic comparisons.
    bool use_switch_nodes = false;

    uint32_t pods = 0;
    uint32_t podsize = 0;
    uint32_t gpus_per_server = 0;
    uint32_t servers_per_tor = 0;
    uint32_t tor_up = 0;
    uint32_t agg_count = 0;
    uint32_t agg_up = 0;

    // Optional overrides (mostly for convenience; runner usually sets these via CLI).
    double link_speed_gbps = 0.0;
    uint32_t link_latency_ns = 0;
    uint32_t switch_latency_ns = 0;

    uint32_t servers_per_pod() const { return (gpus_per_server ? (podsize / gpus_per_server) : 0); }
};

bool load_rail3_config(const char* filename, Rail3Config& out_cfg);
bool load_rail3_config(std::istream& in, Rail3Config& out_cfg);

#endif // RAIL3_CONFIG_H


