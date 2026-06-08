// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "mixnet_config.h"

#include <algorithm>
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

bool load_mixnet_config(const char* filename, MixNetConfig& out_cfg) {
    std::ifstream f(filename);
    if (!f.is_open()) return false;
    return load_mixnet_config(f, out_cfg);
}

bool load_mixnet_config(std::istream& in, MixNetConfig& out_cfg) {
    MixNetConfig cfg;
    // Default: 2 EPS gateways on local GPU 0/1 (MixNet paper-style).
    cfg.eps_gateway_local_indices = {0, 1};

    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> t;
        tokenize_local(line, ' ', t);
        if (is_comment_or_empty(t)) continue;

        if (t[0] == "Type") {
            if (t.size() < 2 || t[1] != "MixNet") {
                std::cerr << "MixNet config: unsupported Type\n";
                return false;
            }
        } else if (t[0] == "GpusPerServer") {
            if (t.size() >= 2) cfg.gpus_per_server = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "EpsGatewayLocal") {
            // Format: EpsGatewayLocal <idx0> <idx1> ...
            cfg.eps_gateway_local_indices.clear();
            for (size_t i = 1; i < t.size(); i++) {
                if (t[i].empty()) continue;
                cfg.eps_gateway_local_indices.push_back((uint32_t)std::stoul(t[i]));
            }
        } else if (t[0] == "EpsTopoFile") {
            if (t.size() >= 2) cfg.eps_topo_file = t[1];
        } else if (t[0] == "OcsScheduleFile") {
            if (t.size() >= 2) cfg.ocs_schedule_file = t[1];
        } else if (t[0] == "OcsPlanes") {
            if (t.size() >= 2) cfg.ocs_planes = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "OcsLinkSpeedGbps") {
            if (t.size() >= 2) cfg.ocs_link_speed_gbps = std::stod(t[1]);
        } else if (t[0] == "OcsQueuePkts") {
            if (t.size() >= 2) cfg.ocs_queue_pkts = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "OcsNoDrop") {
            if (t.size() >= 2) cfg.ocs_no_drop = (std::stoul(t[1]) != 0);
        } else if (t[0] == "OcsLinkLatencyNs") {
            if (t.size() >= 2) cfg.ocs_link_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "OcsSwitchLatencyNs") {
            if (t.size() >= 2) cfg.ocs_switch_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "EpGroupMapFile") {
            if (t.size() >= 2) cfg.ep_group_map_file = t[1];
        } else {
            // Unknown keys ignored for forward compatibility.
            continue;
        }
    }

    if (cfg.gpus_per_server == 0) {
        std::cerr << "MixNet config: missing/invalid GpusPerServer\n";
        return false;
    }
    if (cfg.eps_gateway_local_indices.empty()) {
        std::cerr << "MixNet config: missing/invalid EpsGatewayLocal (need at least 1 index)\n";
        return false;
    }
    for (auto idx : cfg.eps_gateway_local_indices) {
        if (idx >= cfg.gpus_per_server) {
            std::cerr << "MixNet config: EpsGatewayLocal index out of range: " << idx << "\n";
            return false;
        }
    }
    if (cfg.eps_topo_file.empty()) {
        std::cerr << "MixNet config: missing EpsTopoFile\n";
        return false;
    }
    if (cfg.ocs_schedule_file.empty()) {
        std::cerr << "MixNet config: missing OcsScheduleFile\n";
        return false;
    }
    if (cfg.ocs_planes == 0) {
        std::cerr << "MixNet config: missing/invalid OcsPlanes\n";
        return false;
    }
    if (cfg.ep_group_map_file.empty()) {
        std::cerr << "MixNet config: missing EpGroupMapFile\n";
        return false;
    }

    out_cfg = cfg;
    return true;
}

