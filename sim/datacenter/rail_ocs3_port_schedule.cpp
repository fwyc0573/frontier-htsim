// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_ocs3_port_schedule.h"

#include <fstream>
#include <iostream>
#include <sstream>
#include <unordered_set>

static void split_ws(const std::string& line, std::vector<std::string>& out) {
    std::stringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
}

static bool parse_pair_token(const std::string& tok, int32_t& a, int32_t& b) {
    auto pos = tok.find('-');
    if (pos == std::string::npos) return false;
    try {
        a = (int32_t)std::stol(tok.substr(0, pos));
        b = (int32_t)std::stol(tok.substr(pos + 1));
    } catch (...) {
        return false;
    }
    return true;
}

bool load_rail_ocs3_port_schedule_slot0(const std::string& filename,
                                       uint32_t ports,
                                       uint32_t planes,
                                       RailOcs3PortSchedule& out,
                                       bool require_all_planes) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        std::cerr << "OCS schedule: cannot open file: " << filename << "\n";
        return false;
    }
    if (ports == 0 || planes == 0) {
        std::cerr << "OCS schedule: invalid ports/planes (must be > 0)\n";
        return false;
    }

    RailOcs3PortSchedule s;
    s.ports = ports;
    s.planes = planes;
    s.peer.assign(planes, std::vector<int32_t>(ports, -1));

    bool in_slot0 = false;
    std::vector<bool> seen_plane(planes, false);

    std::string line;
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        if (line[0] == '#') continue;
        std::vector<std::string> t;
        split_ws(line, t);
        if (t.empty()) continue;
        if (!t[0].empty() && t[0][0] == '#') continue;

        if (t[0] == "slot") {
            if (t.size() < 2) {
                std::cerr << "OCS schedule: bad slot line\n";
                return false;
            }
            in_slot0 = (t[1] == "0");
            continue;
        }
        if (!in_slot0) continue;

        if (t[0] != "plane") continue;
        if (t.size() < 2) {
            std::cerr << "OCS schedule: bad plane line\n";
            return false;
        }

        uint32_t p = 0;
        try {
            p = (uint32_t)std::stoul(t[1]);
        } catch (...) {
            std::cerr << "OCS schedule: bad plane id\n";
            return false;
        }
        if (p >= planes) {
            std::cerr << "OCS schedule: plane out of range (planes=" << planes << "): " << p << "\n";
            return false;
        }

        // Allow redefining plane: reset its mapping.
        for (uint32_t port = 0; port < ports; port++) s.peer[p][port] = -1;
        seen_plane[p] = true;

        std::unordered_set<uint32_t> used;
        used.reserve(ports);

        for (size_t i = 2; i < t.size(); i++) {
            int32_t a = -1, b = -1;
            if (!parse_pair_token(t[i], a, b)) {
                std::cerr << "OCS schedule: bad pair token " << t[i] << "\n";
                return false;
            }
            if (a < 0 || b < 0) {
                std::cerr << "OCS schedule: negative port id not allowed: " << t[i] << "\n";
                return false;
            }
            if ((uint32_t)a >= ports || (uint32_t)b >= ports) {
                std::cerr << "OCS schedule: port id out of range (ports=" << ports << "): " << t[i] << "\n";
                return false;
            }
            if (a == b) {
                std::cerr << "OCS schedule: self-loop not allowed: " << t[i] << "\n";
                return false;
            }

            uint32_t aa = (uint32_t)a;
            uint32_t bb = (uint32_t)b;
            if (used.count(aa) || used.count(bb)) {
                std::cerr << "OCS schedule: endpoint reused in plane " << p << " token=" << t[i] << "\n";
                return false;
            }
            used.insert(aa);
            used.insert(bb);

            s.peer[p][aa] = (int32_t)bb;
            s.peer[p][bb] = (int32_t)aa;
        }
    }

    if (require_all_planes) {
        for (uint32_t p = 0; p < planes; p++) {
            if (!seen_plane[p]) {
                std::cerr << "OCS schedule: slot 0 missing plane " << p << "\n";
                return false;
            }
        }
    }

    out = s;
    return true;
}


