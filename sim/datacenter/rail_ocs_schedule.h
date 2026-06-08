// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_OCS_SCHEDULE_H
#define RAIL_OCS_SCHEDULE_H

#include <cstdint>
#include <string>
#include <vector>

// Static schedule for RailOCS:
// For each plane p, define a matching between ToRs (rails).
// peer[p][tor] = other_tor if connected in that plane, else -1.
struct RailOcsStaticSchedule {
    uint32_t tors = 0;
    uint32_t planes = 0;
    std::vector<std::vector<int32_t>> peer; // [plane][tor] -> peer tor or -1
};

// Parse a schedule file and extract slot 0 only.
// Expected lines (example):
//   slot 0
//   plane 0 0-1 2-3 ...
bool load_rail_ocs_static_schedule_slot0(const std::string& filename,
                                        uint32_t tors,
                                        uint32_t planes,
                                        RailOcsStaticSchedule& out);

#endif


