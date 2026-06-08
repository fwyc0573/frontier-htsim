// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_ocs_schedule.h"

#include <fstream>
#include <iostream>
#include <sstream>

static void split_ws(const std::string& line, std::vector<std::string>& out) {
    std::stringstream ss(line);
    std::string tok;
    while (ss >> tok) out.push_back(tok);
}

static bool parse_pair(const std::string& tok, int32_t& a, int32_t& b) {
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

bool load_rail_ocs_static_schedule_slot0(const std::string& filename,
                                        uint32_t tors,
                                        uint32_t planes,
                                        RailOcsStaticSchedule& out) {
    std::ifstream f(filename);
    if (!f.is_open()) return false;

    RailOcsStaticSchedule s;
    s.tors = tors;
    s.planes = planes;
    s.peer.assign(planes, std::vector<int32_t>(tors, -1));

    bool in_slot0 = false;
    std::string line;
    while (std::getline(f, line)) {
        // comments
        if (line.empty()) continue;
        if (line[0] == '#') continue;

        std::vector<std::string> t;
        split_ws(line, t);
        if (t.empty()) continue;
        if (t[0][0] == '#') continue;

        if (t[0] == "slot") {
            if (t.size() < 2) {
                std::cerr << "OCS schedule: bad slot line\n";
                return false;
            }
            in_slot0 = (t[1] == "0");
            continue;
        }

        if (!in_slot0) continue;

        if (t[0] == "plane") {
            if (t.size() < 3) {
                std::cerr << "OCS schedule: bad plane line\n";
                return false;
            }
            uint32_t p = (uint32_t)std::stoul(t[1]);
            if (p >= planes) {
                std::cerr << "OCS schedule: plane out of range\n";
                return false;
            }

            // reset plane mapping (allow redefining)
            for (uint32_t tor = 0; tor < tors; tor++) s.peer[p][tor] = -1;

            for (size_t i = 2; i < t.size(); i++) {
                int32_t a = -1, b = -1;
                if (!parse_pair(t[i], a, b)) {
                    std::cerr << "OCS schedule: bad pair token " << t[i] << "\n";
                    return false;
                }
                if (a < 0 || b < 0 || (uint32_t)a >= tors || (uint32_t)b >= tors) {
                    std::cerr << "OCS schedule: ToR index out of range in " << t[i] << "\n";
                    return false;
                }
                if (a == b) {
                    std::cerr << "OCS schedule: self-loop not allowed: " << t[i] << "\n";
                    return false;
                }
                if (s.peer[p][a] != -1 || s.peer[p][b] != -1) {
                    std::cerr << "OCS schedule: ToR appears twice in plane " << p << "\n";
                    return false;
                }
                s.peer[p][a] = b;
                s.peer[p][b] = a;
            }
        }
    }

    out = s;
    return true;
}


