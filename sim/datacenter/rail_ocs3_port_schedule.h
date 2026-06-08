// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
// NOTE: This header is used for the upcoming Rail-OCS3 (3-layer, 2-OCS) design.
// It is intentionally "port-level" (port is an endpoint), which is different from the
// existing 2-layer rail_ocs schedule semantics (ToR id as endpoint).
#ifndef RAIL_OCS3_PORT_SCHEDULE_H
#define RAIL_OCS3_PORT_SCHEDULE_H

#include <cstdint>
#include <string>
#include <vector>

// Rail-OCS3 port-level OCS schedule (slot 0 only).
//
// Semantics:
// - Ports are endpoints (internal OCS ports).
// - For each plane p, peer[p][port] = other_port if connected in that plane, else -1.
//
// File format (compatible with the existing RailOCS schedule style):
//   slot 0
//   plane 0 a-b c-d ...
//   plane 1 ...
//
// Notes:
// - This loader is intentionally generic: it does NOT assume "port id == ToR id".
// - It is meant to be reused by the upcoming Rail-OCS3 design (Agg-OCS + Core-OCS).
struct RailOcs3PortSchedule {
    uint32_t ports = 0;
    uint32_t planes = 0;
    std::vector<std::vector<int32_t>> peer; // [plane][port] -> peer port or -1
};

// Parse a schedule file and extract slot 0 only.
//
// Validation:
// - port ids must be in [0, ports)
// - within a plane, a port may appear at most once
// - plane id must be in [0, planes)
// - if require_all_planes is true, then slot 0 must contain a "plane p ..." line for every p in [0, planes)
//
// Returns true on success, false on parse/validation error.
bool load_rail_ocs3_port_schedule_slot0(const std::string& filename,
                                       uint32_t ports,
                                       uint32_t planes,
                                       RailOcs3PortSchedule& out,
                                       bool require_all_planes = true);

#endif // RAIL_OCS3_PORT_SCHEDULE_H


