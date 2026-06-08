// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_ocs3_config.h"

#include <fstream>
#include <iostream>
#include <algorithm>
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

static std::string to_lower_local(std::string s) {
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return (char)std::tolower(c); });
    return s;
}

bool load_rail_ocs3_config(const char* filename, RailOcs3Config& out_cfg) {
    std::ifstream f(filename);
    if (!f.is_open()) return false;
    return load_rail_ocs3_config(f, out_cfg);
}

bool load_rail_ocs3_config(std::istream& in, RailOcs3Config& out_cfg) {
    RailOcs3Config cfg;
    std::string line;
    while (std::getline(in, line)) {
        std::vector<std::string> t;
        tokenize_local(line, ' ', t);
        if (is_comment_or_empty(t)) continue;

        if (t[0] == "Type") {
            if (t.size() < 2 || t[1] != "RailOCS3") {
                std::cerr << "RailOCS3 config: unsupported Type\n";
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
        } else if (t[0] == "TrunkPortsPerPod") {
            cfg.trunk_ports_per_pod = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "AggOcsPlanes") {
            cfg.agg_ocs_planes = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "CoreOcsPlanes") {
            cfg.core_ocs_planes = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "AggOcsScheduleFile") {
            cfg.agg_ocs_schedule_file = t[1];
        } else if (t[0] == "CoreOcsScheduleFile") {
            cfg.core_ocs_schedule_file = t[1];
        } else if (t[0] == "LinkSpeedGbps") {
            cfg.link_speed_gbps = std::stod(t[1]);
        } else if (t[0] == "OcsQueuePkts") {
            cfg.ocs_queue_pkts = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "OcsNoDrop") {
            cfg.ocs_no_drop = (std::stoul(t[1]) != 0);
        } else if (t[0] == "OcsQueueType") {
            cfg.ocs_queue_type = to_lower_local(t[1]);
        } else if (t[0] == "OcsCutThrough") {
            cfg.ocs_cut_through = (std::stoul(t[1]) != 0);
        } else if (t[0] == "HostFeederBuffer") {
            cfg.host_feeder_buffer = (std::stoul(t[1]) != 0);
        } else if (t[0] == "LinkLatencyNs") {
            cfg.link_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "SwitchLatencyNs") {
            cfg.switch_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "OcsLinkLatencyNs") {
            cfg.ocs_link_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "OcsSwitchLatencyNs") {
            cfg.ocs_switch_latency_ns = (uint32_t)std::stoul(t[1]);
        } else if (t[0] == "SlotUs") {
            cfg.slot_us = std::stod(t[1]);
        } else if (t[0] == "ReconfigUs") {
            cfg.reconfig_us = std::stod(t[1]);
        } else {
            // Unknown keys are ignored to keep the format extensible.
            continue;
        }
    }

    if (cfg.pods == 0) {
        std::cerr << "RailOCS3 config: missing Pods\n";
        return false;
    }
    if (cfg.podsize == 0) {
        std::cerr << "RailOCS3 config: missing Podsize\n";
        return false;
    }
    if (cfg.gpus_per_server == 0) {
        std::cerr << "RailOCS3 config: missing/invalid GpusPerServer\n";
        return false;
    }
    if (cfg.podsize % cfg.gpus_per_server != 0) {
        std::cerr << "RailOCS3 config: Podsize must be multiple of GpusPerServer\n";
        return false;
    }
    if (cfg.tor_up == 0) {
        std::cerr << "RailOCS3 config: missing TorUp\n";
        return false;
    }
    if (cfg.trunk_ports_per_pod == 0) {
        std::cerr << "RailOCS3 config: missing TrunkPortsPerPod\n";
        return false;
    }
    if (cfg.agg_ocs_planes == 0) {
        std::cerr << "RailOCS3 config: missing AggOcsPlanes\n";
        return false;
    }
    if (cfg.core_ocs_planes == 0) {
        std::cerr << "RailOCS3 config: missing CoreOcsPlanes\n";
        return false;
    }
    if (cfg.agg_ocs_schedule_file.empty()) {
        std::cerr << "RailOCS3 config: missing AggOcsScheduleFile\n";
        return false;
    }
    if (cfg.core_ocs_schedule_file.empty()) {
        std::cerr << "RailOCS3 config: missing CoreOcsScheduleFile\n";
        return false;
    }
    if (cfg.link_speed_gbps <= 0.0) {
        std::cerr << "RailOCS3 config: missing LinkSpeedGbps\n";
        return false;
    }
    if (!cfg.ocs_queue_type.empty()) {
        std::string qt = to_lower_local(cfg.ocs_queue_type);
        if (!(qt == "fifo" || qt == "composite" || qt == "random")) {
            std::cerr << "RailOCS3 config: invalid OcsQueueType (supported: fifo|composite|random)\n";
            return false;
        }
        cfg.ocs_queue_type = qt;
    }
    if (cfg.slot_us <= 0.0) {
        std::cerr << "RailOCS3 config: missing SlotUs\n";
        return false;
    }
    if (cfg.reconfig_us < 0.0) {
        std::cerr << "RailOCS3 config: invalid ReconfigUs\n";
        return false;
    }

    // servers_per_tor defaults to servers_per_pod (no split).
    if (cfg.servers_per_tor == 0) {
        cfg.servers_per_tor = cfg.servers_per_pod();
    }
    if (cfg.servers_per_tor == 0 || cfg.servers_per_tor > cfg.servers_per_pod()) {
        std::cerr << "RailOCS3 config: invalid ServersPerTor (per pod)\n";
        return false;
    }

    out_cfg = cfg;
    return true;
}


