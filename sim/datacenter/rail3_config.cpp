// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail3_config.h"

#include <algorithm>
#include <cctype>
#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

static void tokenize_local(std::string const& str, char delim, std::vector<std::string>& out) {
    std::stringstream ss(str);
    std::string s;
    while (std::getline(ss, s, delim)) out.push_back(s);
}

static bool is_comment_or_empty(const std::vector<std::string>& t) {
    if (t.empty()) return true;
    if (t[0].empty()) return true;
    return t[0][0] == '#';
}

bool load_rail3_config(const char* filename, Rail3Config& out_cfg) {
    std::ifstream f(filename);
    if (!f.is_open()) return false;
    return load_rail3_config(f, out_cfg);
}

bool load_rail3_config(std::istream& in, Rail3Config& out_cfg) {
    Rail3Config cfg;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> t;
        tokenize_local(line, ' ', t);
        if (is_comment_or_empty(t)) continue;

        if (t[0] == "Type") {
            if (t.size() < 2) {
                std::cerr << "Rail3 config: Type requires a value\n";
                return false;
            }
            if (t[1] == "Rail3") {
                cfg.mode = Rail3Config::RAIL3;
            } else if (t[1] == "Rail3FT") {
                cfg.mode = Rail3Config::RAIL3_FTLIKE;
            } else {
                std::cerr << "Rail3 config: unsupported Type (expected Rail3 or Rail3FT)\n";
                return false;
            }
        } else if (t[0] == "Pods") {
            cfg.pods = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "Podsize") {
            cfg.podsize = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "GpusPerServer") {
            cfg.gpus_per_server = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "ServersPerTor") {
            cfg.servers_per_tor = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "TorUp") {
            cfg.tor_up = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "AggCount") {
            cfg.agg_count = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "AggUp") {
            cfg.agg_up = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "UseSwitchNodes") {
            cfg.use_switch_nodes = (std::stoul(t[1]) != 0);
        } else if (t[0] == "LinkSpeedGbps") {
            cfg.link_speed_gbps = std::stod(t[1]);
        } else if (t[0] == "LinkLatencyNs") {
            cfg.link_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "SwitchLatencyNs") {
            cfg.switch_latency_ns = (uint32_t)std::stoul(t[1]);
        } else {
            // Unknown keys are ignored to keep the format extensible.
            continue;
        }
    }

    if (cfg.pods == 0) {
        std::cerr << "Rail3 config: missing Pods\n";
        return false;
    }
    if (cfg.podsize == 0) {
        std::cerr << "Rail3 config: missing Podsize\n";
        return false;
    }
    if (cfg.gpus_per_server == 0) {
        std::cerr << "Rail3 config: missing/invalid GpusPerServer\n";
        return false;
    }
    if (cfg.podsize % cfg.gpus_per_server != 0) {
        std::cerr << "Rail3 config: Podsize must be multiple of GpusPerServer\n";
        return false;
    }
    if (cfg.tor_up == 0) {
        std::cerr << "Rail3 config: missing TorUp\n";
        return false;
    }

    // servers_per_tor defaults to servers_per_pod (no split).
    if (cfg.servers_per_tor == 0) {
        cfg.servers_per_tor = cfg.servers_per_pod();
    }
    if (cfg.servers_per_tor == 0 || cfg.servers_per_tor > cfg.servers_per_pod()) {
        std::cerr << "Rail3 config: invalid ServersPerTor (per pod)\n";
        return false;
    }

    // If not specified:
    // - AggCount defaults to TorUp (each ToR connects to all Aggs with 1 link).
    // - AggUp defaults to tors_per_pod (so each Agg has same up/down radix => NB within a pod).
    if (cfg.agg_count == 0) cfg.agg_count = cfg.tor_up;
    // Current Rail3Topology implementation models ToR->Agg as a full mesh over AggCount.
    // To avoid silently ignoring TorUp, require AggCount == TorUp.
    if (cfg.agg_count != cfg.tor_up) {
        std::cerr << "Rail3 config: AggCount must equal TorUp in current implementation (got AggCount="
                  << cfg.agg_count << " TorUp=" << cfg.tor_up << ")\n";
        return false;
    }
    uint32_t tors_per_pod = cfg.gpus_per_server * ((cfg.servers_per_pod() + cfg.servers_per_tor - 1) / cfg.servers_per_tor);
    if (tors_per_pod == 0) tors_per_pod = cfg.gpus_per_server;
    if (cfg.agg_up == 0) cfg.agg_up = tors_per_pod;

    out_cfg = cfg;
    return true;
}


