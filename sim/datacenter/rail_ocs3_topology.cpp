// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_ocs3_topology.h"

#include "randomqueue.h"
#include "compositequeue.h"
#include "main.h" // for FEEDER_BUFFER

#include <cassert>
#include <fstream>
#include <iostream>
#include <sstream>

static std::string u32s(uint32_t v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

bool RailOcs3Topology::has_intra_pod_direct_tor_circuit(uint32_t pod,
                                                       uint32_t tor_a_in_pod,
                                                       uint32_t tor_b_in_pod,
                                                       uint32_t* out_plane) const {
    if (pod >= _pods) return false;
    if (tor_a_in_pod >= _tors_per_pod || tor_b_in_pod >= _tors_per_pod) return false;
    for (uint32_t p = 0; p < _agg_ocs_planes; p++) {
        int32_t peer = _tor_to_tor[pod][p][tor_a_in_pod];
        if (peer < 0) continue;
        if ((uint32_t)peer == tor_b_in_pod) {
            if (out_plane) *out_plane = p;
            return true;
        }
    }
    return false;
}

bool RailOcs3Topology::has_src_tor_to_any_trunk(uint32_t pod,
                                               uint32_t tor_in_pod,
                                               uint32_t* out_plane,
                                               uint32_t* out_trunk) const {
    if (pod >= _pods) return false;
    if (tor_in_pod >= _tors_per_pod) return false;
    for (uint32_t p = 0; p < _agg_ocs_planes; p++) {
        int32_t trunk = _tor_to_trunk[pod][p][tor_in_pod];
        if (trunk < 0) continue;
        if (out_plane) *out_plane = p;
        if (out_trunk) *out_trunk = (uint32_t)trunk;
        return true;
    }
    return false;
}

bool RailOcs3Topology::has_dst_trunk_to_tor(uint32_t pod,
                                           uint32_t trunk,
                                           uint32_t tor_in_pod,
                                           uint32_t* out_plane) const {
    if (pod >= _pods) return false;
    if (trunk >= _trunk_ports_per_pod) return false;
    if (tor_in_pod >= _tors_per_pod) return false;
    for (uint32_t p = 0; p < _agg_ocs_planes; p++) {
        int32_t t = _trunk_to_tor[pod][p][trunk];
        if (t < 0) continue;
        if ((uint32_t)t == tor_in_pod) {
            if (out_plane) *out_plane = p;
            return true;
        }
    }
    return false;
}

bool RailOcs3Topology::has_core_trunk_to_pod(uint32_t src_pod,
                                            uint32_t src_trunk,
                                            uint32_t dst_pod,
                                            uint32_t* out_plane,
                                            uint32_t* out_dst_trunk) const {
    if (src_pod >= _pods || dst_pod >= _pods) return false;
    if (src_trunk >= _trunk_ports_per_pod) return false;
    uint32_t src_gt = src_pod * _trunk_ports_per_pod + src_trunk;
    for (uint32_t p = 0; p < _core_ocs_planes; p++) {
        int32_t peer = _core_sched.peer[p][src_gt];
        if (peer < 0) continue;
        uint32_t dst_gt = (uint32_t)peer;
        uint32_t peer_pod = dst_gt / _trunk_ports_per_pod;
        uint32_t peer_trunk = dst_gt % _trunk_ports_per_pod;
        if (peer_pod != dst_pod) continue;
        if (out_plane) *out_plane = p;
        if (out_dst_trunk) *out_dst_trunk = peer_trunk;
        return true;
    }
    return false;
}

static void free_paths(std::vector<const Route*>* paths) {
    if (!paths) return;
    for (auto* r : *paths) delete r;
    delete paths;
}

// A FIFO that never drops packets due to buffer overflow.
class NoDropQueue final : public Queue {
public:
    NoDropQueue(linkspeed_bps bitrate, mem_b maxsize, EventList& eventlist, QueueLogger* logger)
        : Queue(bitrate, maxsize, eventlist, logger) {}

    void receivePacket(Packet& pkt) override {
        pkt.flow().logTraffic(pkt, *this, TrafficLogger::PKT_ARRIVE);
        bool queueWasEmpty = _enqueued.empty();
        Packet* pkt_p = &pkt;
        _enqueued.push(pkt_p);
        _queuesize += pkt.size();
        if (_logger) _logger->logQueue(*this, QueueLogger::PKT_ENQUEUE, pkt);
        if (queueWasEmpty) {
            assert(_enqueued.size() == 1);
            beginService();
        }
    }
};

RailOcs3Topology::RailOcs3Topology(const RailOcs3Config& cfg,
                                 const RailOcs3PortSchedule& agg_ocs_sched,
                                 const RailOcs3PortSchedule& core_ocs_sched,
                                 linkspeed_bps linkspeed,
                                 mem_b switch_queuesize_bytes,
                                 mem_b ocs_queuesize_bytes,
                                 simtime_picosec link_latency,
                                 simtime_picosec switch_latency,
                                 simtime_picosec ocs_link_latency,
                                 simtime_picosec ocs_switch_latency,
                                 bool ocs_no_drop,
                                 queue_type switch_queue_type,
                                 queue_type host_queue_type,
                                 Logfile* logfile,
                                 EventList* eventlist)
    : _cfg(cfg),
      _no_of_nodes(cfg.nodes()),
      _pods(cfg.pods),
      _podsize(cfg.podsize),
      _gpus_per_server(cfg.gpus_per_server),
      _servers_per_tor(cfg.servers_per_tor ? cfg.servers_per_tor : cfg.servers_per_pod()),
      _rail_splits_per_pod(cfg.rail_splits_per_pod()),
      _tors_per_pod(cfg.tors_per_pod()),
      _tors_total(cfg.pods * cfg.tors_per_pod()),
      _tor_up(cfg.tor_up),
      _trunk_ports_per_pod(cfg.trunk_ports_per_pod),
      _agg_ocs_ports_per_pod(cfg.agg_ocs_ports_per_pod()),
      _agg_ocs_planes(cfg.agg_ocs_planes),
      _core_ocs_planes(cfg.core_ocs_planes),
      _core_ocs_ports_total(cfg.core_ocs_ports_total()),
      _agg_sched(agg_ocs_sched),
      _core_sched(core_ocs_sched),
      _linkspeed(linkspeed),
      _switch_queuesize_bytes(switch_queuesize_bytes),
      _ocs_queuesize_bytes(ocs_queuesize_bytes),
      _link_latency(link_latency),
      _switch_latency(switch_latency),
      _ocs_link_latency(ocs_link_latency),
      _ocs_switch_latency(ocs_switch_latency),
      _ocs_no_drop(ocs_no_drop),
      _ocs_cut_through(cfg.ocs_cut_through),
      _host_feeder_buffer(cfg.host_feeder_buffer),
      _switch_queue_type(switch_queue_type),
      _host_queue_type(host_queue_type),
      _logfile(logfile),
      _eventlist(eventlist) {
    assert(_pods > 0);
    assert(_podsize > 0);
    assert(_gpus_per_server > 0);
    assert(_tors_per_pod > 0);
    assert(_tor_up > 0);
    assert(_trunk_ports_per_pod > 0);
    assert(_agg_ocs_planes > 0);
    assert(_core_ocs_planes > 0);
    assert(_agg_sched.planes == _agg_ocs_planes);
    assert(_core_sched.planes == _core_ocs_planes);
    // Port-count sanity checks.
    assert(_agg_sched.ports == _agg_ocs_ports_per_pod);
    assert(_core_sched.ports == _core_ocs_ports_total);

    init_host_tor_edges();
    init_agg_ocs_edges();
    init_core_ocs_edges();
}

uint32_t RailOcs3Topology::tor_in_pod_of_host(uint32_t host) const {
    uint32_t pod = pod_of_host(host);
    (void)pod;
    uint32_t h = local_host(host);
    uint32_t rail = rail_of_local_host(h);
    uint32_t srv = server_of_local_host(h);
    uint32_t group = srv / _servers_per_tor;
    if (group >= _rail_splits_per_pod) group = _rail_splits_per_pod - 1;
    return rail * _rail_splits_per_pod + group;
}

HostQueue* RailOcs3Topology::make_host_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    HostQueue* q = nullptr;
    mem_b host_q = _host_feeder_buffer ? memFromPkt(FEEDER_BUFFER) : _switch_queuesize_bytes;
    if (_host_queue_type == PRIORITY) {
        q = new PriorityQueue(_linkspeed, host_q, *_eventlist, ql);
    } else {
        q = new FairPriorityQueue(_linkspeed, host_q, *_eventlist, ql);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* RailOcs3Topology::make_switch_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    BaseQueue* q = nullptr;
    if (_switch_queue_type == COMPOSITE || _switch_queue_type == COMPOSITE_ECN || _switch_queue_type == COMPOSITE_ECN_LB) {
        q = new CompositeQueue(_linkspeed, _switch_queuesize_bytes, *_eventlist, ql);
    } else {
        q = new RandomQueue(_linkspeed, _switch_queuesize_bytes, *_eventlist, ql, 0);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* RailOcs3Topology::make_ocs_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    BaseQueue* q = nullptr;
    if (_ocs_no_drop) {
        q = new NoDropQueue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
    } else {
        // OCS queue type is configurable; default is legacy FIFO Queue.
        // Note: CompositeQueue implements "trim" semantics on overflow (strip_payload), which tends to
        // better match NDP-style modeling than pure drop FIFO under heavy incast/outcast.
        std::string qt = _cfg.ocs_queue_type;
        if (qt.empty() || qt == "fifo") {
            q = new Queue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
        } else if (qt == "composite") {
            q = new CompositeQueue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
        } else if (qt == "random") {
            q = new RandomQueue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql, 0);
        } else {
            // Defensive fallback (should be validated by config loader).
            q = new Queue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
        }
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* RailOcs3Topology::maybe_make_ocs_queue(const std::string& name, bool tor_egress) {
    if (_ocs_cut_through) {
        // Cut-through modeling (bandwidth-preserving):
        // - Keep serialization/queueing at the transmitter (ToR uplink egress).
        // - Remove internal OCS/trunk queues to avoid adding store-and-forward hops.
        if (!tor_egress) {
            (void)name;
            return nullptr;
        }
        return make_ocs_queue(name);
    }
    return make_ocs_queue(name);
}

Pipe* RailOcs3Topology::make_pipe(const std::string& name) {
    auto* p = new Pipe(_link_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

Pipe* RailOcs3Topology::make_ocs_pipe(const std::string& name) {
    auto* p = new Pipe(_ocs_link_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

void RailOcs3Topology::init_host_tor_edges() {
    _host_to_tor.resize(_no_of_nodes);
    _tor_to_host.assign(_tors_total, std::vector<Edge>(_no_of_nodes));

    for (uint32_t h = 0; h < _no_of_nodes; h++) {
        uint32_t pod = pod_of_host(h);
        uint32_t tor_in_pod = tor_in_pod_of_host(h);
        uint32_t tor = tor_global(pod, tor_in_pod);
        _host_to_tor[h].q = make_host_queue("H" + u32s(h) + "->ToR" + u32s(tor));
        _host_to_tor[h].p = make_pipe("P_H" + u32s(h) + "_ToR" + u32s(tor));

        _tor_to_host[tor][h].q = make_switch_queue("ToR" + u32s(tor) + "->H" + u32s(h));
        _tor_to_host[tor][h].p = make_pipe("P_ToR" + u32s(tor) + "_H" + u32s(h));
    }
}

void RailOcs3Topology::init_agg_ocs_edges() {
    // =========================
    // Agg-OCS parsing (LIMITATION: ToR-level compression)
    // =========================
    // The Agg-OCS schedule is port-level:
    //   - ports [0 .. tors_per_pod*tor_up) are ToR uplink ports (one endpoint per uplink),
    //   - ports [tors_per_pod*tor_up ..) are trunk ports.
    //
    // However, the current implementation compresses the schedule into ToR-level peers per plane:
    //   - per (pod, plane), each ToR is allowed at most ONE trunk peer (ToR->trunk), and
    //     at most ONE ToR peer (ToR->ToR).
    // If the schedule uses multiple uplinks of the same ToR in the same plane (legal at port-level),
    // we currently reject (assert) to keep route enumeration simple.

    // Per pod schedule uses local port ids:
    // [0 .. tors*tor_up) are ToR uplink ports, [tors*tor_up ..) are trunk ports.
    uint32_t tor_ports = _tors_per_pod * _tor_up;
    assert(tor_ports + _trunk_ports_per_pod == _agg_ocs_ports_per_pod);

    _tor_to_trunk.assign(_pods, std::vector<std::vector<int32_t>>(_agg_ocs_planes, std::vector<int32_t>(_tors_per_pod, -1)));
    _trunk_to_tor.assign(_pods, std::vector<std::vector<int32_t>>(_agg_ocs_planes, std::vector<int32_t>(_trunk_ports_per_pod, -1)));
    _agg_tor_to_trunk.assign(_pods, std::vector<std::vector<Edge>>(_agg_ocs_planes, std::vector<Edge>(_tors_per_pod)));
    _agg_trunk_to_tor.assign(_pods, std::vector<std::vector<Edge>>(_agg_ocs_planes, std::vector<Edge>(_trunk_ports_per_pod)));
    _tor_to_tor.assign(_pods, std::vector<std::vector<int32_t>>(_agg_ocs_planes, std::vector<int32_t>(_tors_per_pod, -1)));
    _agg_tor_to_tor.assign(_pods, std::vector<std::vector<Edge>>(_agg_ocs_planes, std::vector<Edge>(_tors_per_pod)));

    // Build mapping and instantiate queues/pipes for each plane.
    for (uint32_t pod = 0; pod < _pods; pod++) {
        (void)pod;
        for (uint32_t p = 0; p < _agg_ocs_planes; p++) {
            // Scan all ToR uplink ports; if a port is paired to a trunk, record per-ToR mapping.
            for (uint32_t port = 0; port < tor_ports; port++) {
                int32_t peer = _agg_sched.peer[p][port];
                if (peer < 0) continue;
                if ((uint32_t)peer < tor_ports) {
                        // ToR<->ToR in Agg-OCS.
                        uint32_t tor_a = port / _tor_up;
                        uint32_t tor_b = (uint32_t)peer / _tor_up;
                        if (tor_a >= _tors_per_pod || tor_b >= _tors_per_pod) continue;
                        if (tor_a == tor_b) continue;
                        // LIMITATION: at most one ToR<->ToR peer per ToR per plane.
                        if (_tor_to_tor[pod][p][tor_a] != -1 && (uint32_t)_tor_to_tor[pod][p][tor_a] != tor_b) {
                            std::cerr << "RailOCS3 Agg-OCS: multiple ToR peers for ToR " << tor_a << " in plane " << p << "\n";
                            assert(false);
                        }
                        _tor_to_tor[pod][p][tor_a] = (int32_t)tor_b;
                        // Note: the reverse mapping will be seen when scanning the peer port, but we set it too.
                        if (_tor_to_tor[pod][p][tor_b] != -1 && (uint32_t)_tor_to_tor[pod][p][tor_b] != tor_a) {
                            std::cerr << "RailOCS3 Agg-OCS: multiple ToR peers for ToR " << tor_b << " in plane " << p << "\n";
                            assert(false);
                        }
                        _tor_to_tor[pod][p][tor_b] = (int32_t)tor_a;
                        continue;
                }
                uint32_t trunk = (uint32_t)peer - tor_ports;
                uint32_t tor = port / _tor_up;
                if (tor >= _tors_per_pod || trunk >= _trunk_ports_per_pod) continue;

                // LIMITATION: at most one uplink per ToR per plane.
                if (_tor_to_trunk[pod][p][tor] != -1) {
                    // If multiple ports of the same ToR appear in one plane, we currently reject to keep semantics simple.
                    // This can be generalized later by treating ToR uplinks as distinct endpoints in route enumeration.
                    std::cerr << "RailOCS3 Agg-OCS: multiple uplinks for ToR " << tor << " in plane " << p << "\n";
                    assert(false);
                }
                _tor_to_trunk[pod][p][tor] = (int32_t)trunk;
                _trunk_to_tor[pod][p][trunk] = (int32_t)tor;
            }

            // Instantiate OCS queues for each ToR<->trunk pairing in this plane.
            for (uint32_t tor = 0; tor < _tors_per_pod; tor++) {
                int32_t trunk_i = _tor_to_trunk[pod][p][tor];
                if (trunk_i < 0) continue;
                uint32_t trunk = (uint32_t)trunk_i;
                // Create directed edges ToR->trunk and trunk->ToR.
                uint32_t tor_g = tor_global(pod, tor);
                auto name_fwd = "AggOCS p" + u32s(p) + " ToR" + u32s(tor_g) + "->Trunk" + u32s(pod) + ":" + u32s(trunk);
                auto name_rev = "AggOCS p" + u32s(p) + " Trunk" + u32s(pod) + ":" + u32s(trunk) + "->ToR" + u32s(tor_g);
                _agg_tor_to_trunk[pod][p][tor].q = maybe_make_ocs_queue(name_fwd, /*tor_egress=*/true);
                _agg_tor_to_trunk[pod][p][tor].p = make_ocs_pipe("P_" + name_fwd);
                _agg_trunk_to_tor[pod][p][trunk].q = maybe_make_ocs_queue(name_rev, /*tor_egress=*/false);
                _agg_trunk_to_tor[pod][p][trunk].p = make_ocs_pipe("P_" + name_rev);
            }

                // Instantiate OCS queues for each ToR<->ToR pairing in this plane.
                // We create a directed edge per ToR that has a peer.
                for (uint32_t tor = 0; tor < _tors_per_pod; tor++) {
                    int32_t peer_tor_i = _tor_to_tor[pod][p][tor];
                    if (peer_tor_i < 0) continue;
                    uint32_t peer_tor = (uint32_t)peer_tor_i;
                    if (peer_tor >= _tors_per_pod) continue;
                    uint32_t tor_g = tor_global(pod, tor);
                    uint32_t peer_g = tor_global(pod, peer_tor);
                    auto name = "AggOCS p" + u32s(p) + " ToR" + u32s(tor_g) + "->ToR" + u32s(peer_g);
                    _agg_tor_to_tor[pod][p][tor].q = maybe_make_ocs_queue(name, /*tor_egress=*/true);
                    _agg_tor_to_tor[pod][p][tor].p = make_ocs_pipe("P_" + name);
                }
        }
    }
}

void RailOcs3Topology::init_core_ocs_edges() {
    _core_link.assign(_core_ocs_planes, std::vector<Edge>(_core_ocs_ports_total));

    for (uint32_t p = 0; p < _core_ocs_planes; p++) {
        for (uint32_t a = 0; a < _core_ocs_ports_total; a++) {
            int32_t b = _core_sched.peer[p][a];
            if (b < 0) continue;
            uint32_t bb = (uint32_t)b;
            if (a < bb) {
                auto name_ab = "CoreOCS p" + u32s(p) + " Trunk" + u32s(a) + "->" + u32s(bb);
                auto name_ba = "CoreOCS p" + u32s(p) + " Trunk" + u32s(bb) + "->" + u32s(a);
                _core_link[p][a].q = maybe_make_ocs_queue(name_ab, /*tor_egress=*/false);
                _core_link[p][a].p = make_ocs_pipe("P_" + name_ab);
                _core_link[p][bb].q = maybe_make_ocs_queue(name_ba, /*tor_egress=*/false);
                _core_link[p][bb].p = make_ocs_pipe("P_" + name_ba);
            }
        }
    }
}

vector<const Route*>* RailOcs3Topology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    (void)reverse;
    auto* paths = new vector<const Route*>();
    assert(src < _no_of_nodes && dest < _no_of_nodes);
    assert(src != dest);

    uint32_t src_pod = pod_of_host(src);
    uint32_t dst_pod = pod_of_host(dest);
    uint32_t src_tor_in_pod = tor_in_pod_of_host(src);
    uint32_t dst_tor_in_pod = tor_in_pod_of_host(dest);
    uint32_t src_tor = tor_global(src_pod, src_tor_in_pod);
    uint32_t dst_tor = tor_global(dst_pod, dst_tor_in_pod);

    // Same ToR: direct.
    if (src_tor == dst_tor) {
        auto* r = new Route();
        r->push_back(_host_to_tor[src].q);
        r->push_back(_host_to_tor[src].p);
        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(src_tor));
            r->push_back(sw);
        }
        r->push_back(_tor_to_host[dst_tor][dest].q);
        r->push_back(_tor_to_host[dst_tor][dest].p);
        paths->push_back(r);
        return paths;
    }

    // Same pod, different ToR (LIMITATION: only direct ToR<->ToR circuits supported):
    // - We ONLY support direct ToR<->ToR circuits via Agg-OCS schedule.
    // - Multi-hop intra-pod paths (e.g., via trunk/Core-OCS) are not implemented.
    if (src_pod == dst_pod) {
        for (uint32_t p = 0; p < _agg_ocs_planes; p++) {
            int32_t peer_tor_i = _tor_to_tor[src_pod][p][src_tor_in_pod];
            if (peer_tor_i < 0) continue;
            if ((uint32_t)peer_tor_i != dst_tor_in_pod) continue;

            auto* r = new Route();
            r->push_back(_host_to_tor[src].q);
            r->push_back(_host_to_tor[src].p);

            if (_ocs_switch_latency) {
                auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
                sw->setName("SwLat_AggOCS_p" + u32s(p));
                r->push_back(sw);
            }
            if (_agg_tor_to_tor[src_pod][p][src_tor_in_pod].q) r->push_back(_agg_tor_to_tor[src_pod][p][src_tor_in_pod].q);
            r->push_back(_agg_tor_to_tor[src_pod][p][src_tor_in_pod].p);

            r->push_back(_tor_to_host[dst_tor][dest].q);
            r->push_back(_tor_to_host[dst_tor][dest].p);
            paths->push_back(r);
        }
        return paths;
    }

    // Cross-pod: enumerate paths via two OCS fabrics:
    //   ToR(src) -> Agg-OCS(ToR->trunk) -> Core-OCS(trunk->trunk) -> Agg-OCS(trunk->ToR) -> ToR(dst)
    for (uint32_t p_src = 0; p_src < _agg_ocs_planes; p_src++) {
        int32_t trunk_i = _tor_to_trunk[src_pod][p_src][src_tor_in_pod];
        if (trunk_i < 0) continue;
        uint32_t trunk = (uint32_t)trunk_i;
        uint32_t src_gt = src_pod * _trunk_ports_per_pod + trunk;

        for (uint32_t p_core = 0; p_core < _core_ocs_planes; p_core++) {
            int32_t dst_gt_i = _core_sched.peer[p_core][src_gt];
            if (dst_gt_i < 0) continue;
            uint32_t dst_gt = (uint32_t)dst_gt_i;
            uint32_t peer_pod = dst_gt / _trunk_ports_per_pod;
            uint32_t peer_trunk = dst_gt % _trunk_ports_per_pod;
            if (peer_pod != dst_pod) continue;

            for (uint32_t p_dst = 0; p_dst < _agg_ocs_planes; p_dst++) {
                int32_t tor2_i = _trunk_to_tor[dst_pod][p_dst][peer_trunk];
                if (tor2_i < 0) continue;
                if ((uint32_t)tor2_i != dst_tor_in_pod) continue;

                // Build route
                auto* r = new Route();
                r->push_back(_host_to_tor[src].q);
                r->push_back(_host_to_tor[src].p);

                if (_ocs_switch_latency) {
                    auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
                    sw->setName("SwLat_AggOCS_src_p" + u32s(p_src));
                    r->push_back(sw);
                }
                if (_agg_tor_to_trunk[src_pod][p_src][src_tor_in_pod].q) r->push_back(_agg_tor_to_trunk[src_pod][p_src][src_tor_in_pod].q);
                r->push_back(_agg_tor_to_trunk[src_pod][p_src][src_tor_in_pod].p);

                if (_ocs_switch_latency) {
                    auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
                    sw->setName("SwLat_CoreOCS_p" + u32s(p_core));
                    r->push_back(sw);
                }
                if (_core_link[p_core][src_gt].q) r->push_back(_core_link[p_core][src_gt].q);
                r->push_back(_core_link[p_core][src_gt].p);

                if (_ocs_switch_latency) {
                    auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
                    sw->setName("SwLat_AggOCS_dst_p" + u32s(p_dst));
                    r->push_back(sw);
                }
                if (_agg_trunk_to_tor[dst_pod][p_dst][peer_trunk].q) r->push_back(_agg_trunk_to_tor[dst_pod][p_dst][peer_trunk].q);
                r->push_back(_agg_trunk_to_tor[dst_pod][p_dst][peer_trunk].p);

                r->push_back(_tor_to_host[dst_tor][dest].q);
                r->push_back(_tor_to_host[dst_tor][dest].p);
                paths->push_back(r);
            }
        }
    }

    return paths;
}

vector<uint32_t>* RailOcs3Topology::get_neighbours(uint32_t src) {
    auto* n = new vector<uint32_t>();
    if (src >= _no_of_nodes) return n;
    // Conservative: neighbors unknown for now; used by some traffic generators only.
    return n;
}

void RailOcs3Topology::dump_reachability(const std::string& filename,
                                        uint32_t max_unreachable_examples,
                                        uint32_t max_tor_pairs_print) {
    std::ofstream out(filename.c_str());
    if (!out) {
        std::cerr << "RailOCS3: failed to open reachability dump file for writing: " << filename << "\n";
        return;
    }

    out << "RailOCS3 reachability dump (derived from get_bidir_paths() current routing logic)\n";
    out << "nodes=" << _no_of_nodes
        << " pods=" << _pods
        << " podsize=" << _podsize
        << " gpus_per_server=" << _gpus_per_server
        << " tors_per_pod=" << _tors_per_pod
        << " tor_up=" << _tor_up
        << " trunk_ports_per_pod=" << _trunk_ports_per_pod
        << " agg_planes=" << _agg_ocs_planes
        << " core_planes=" << _core_ocs_planes
        << "\n\n";

    // Pick one representative host per global ToR by scanning all hosts.
    std::vector<int32_t> rep_host(_tors_total, -1);
    for (uint32_t h = 0; h < _no_of_nodes; h++) {
        uint32_t pod = pod_of_host(h);
        uint32_t tor_in_pod = tor_in_pod_of_host(h);
        uint32_t tor_g = tor_global(pod, tor_in_pod);
        if (tor_g < _tors_total && rep_host[tor_g] < 0) rep_host[tor_g] = (int32_t)h;
    }
    uint32_t missing_rep = 0;
    for (uint32_t t = 0; t < _tors_total; t++) if (rep_host[t] < 0) missing_rep++;
    if (missing_rep) {
        out << "WARNING: missing representative hosts for " << missing_rep << " ToRs; ToR-level matrix may be incomplete.\n\n";
    }

    out << "## ToR-level reachability matrix (1=reachable, 0=unreachable, ?=not evaluated due to cap)\n";
    out << "## Rows/cols are global ToR IDs. Diagonal is always 1.\n";
    out << "## Note: reachability is computed using representative hosts and get_bidir_paths().\n\n";

    uint32_t tor_pairs_checked = 0;
    uint32_t tor_pairs_reachable = 0;
    uint32_t tor_pairs_unreachable = 0;

    out << "tor_id";
    for (uint32_t t = 0; t < _tors_total; t++) out << "\t" << t;
    out << "\n";

    for (uint32_t a = 0; a < _tors_total; a++) {
        out << a;
        for (uint32_t b = 0; b < _tors_total; b++) {
            if (a == b) {
                out << "\t1";
                continue;
            }
            if (rep_host[a] < 0 || rep_host[b] < 0) {
                out << "\t0";
                continue;
            }
            if (tor_pairs_checked < max_tor_pairs_print) {
                auto* paths = get_bidir_paths((uint32_t)rep_host[a], (uint32_t)rep_host[b], false);
                bool ok = (paths && !paths->empty());
                free_paths(paths);
                if (ok) tor_pairs_reachable++; else tor_pairs_unreachable++;
                tor_pairs_checked++;
                out << "\t" << (ok ? "1" : "0");
            } else {
                out << "\t?";
            }
        }
        out << "\n";
    }

    out << "\n## ToR-level summary (only for the first " << tor_pairs_checked << " evaluated non-diagonal pairs)\n";
    out << "reachable=" << tor_pairs_reachable << " unreachable=" << tor_pairs_unreachable << "\n\n";

    // =========================
    // ToR-pair "bandwidth" view
    // =========================
    // We report (path_count, bw_gbps) where:
    // - intra-pod: path_count = number of Agg-OCS planes that provide a direct ToR<->ToR circuit.
    // - cross-pod: path_count = number of (p_src, p_core, p_dst) triples that produce a valid route:
    //       ToR(src) -> trunk -> (core trunk->trunk into dst pod) -> trunk -> ToR(dst)
    //
    // This is a convenient debug proxy, NOT a full max-flow computation:
    // - It ignores contention with other ToRs/flows.
    // - It treats each enumerated path as providing one full-rate circuit of _linkspeed.
    //
    double link_gbps = (double)_linkspeed / 1e9;
    out << "## ToR-pair capacity proxy (path_count and bw_gbps = path_count * linkspeed)\n";
    out << "## linkspeed_gbps=" << link_gbps << "\n\n";

    auto intra_path_count = [&](uint32_t pod, uint32_t a_in_pod, uint32_t b_in_pod) -> uint32_t {
        if (pod >= _pods) return 0;
        if (a_in_pod >= _tors_per_pod || b_in_pod >= _tors_per_pod) return 0;
        if (a_in_pod == b_in_pod) return 0;
        uint32_t cnt = 0;
        for (uint32_t p = 0; p < _agg_ocs_planes; p++) {
            int32_t peer = _tor_to_tor[pod][p][a_in_pod];
            if (peer >= 0 && (uint32_t)peer == b_in_pod) cnt++;
        }
        return cnt;
    };

    auto cross_path_count = [&](uint32_t sp, uint32_t st_in_pod, uint32_t dp, uint32_t dt_in_pod) -> uint32_t {
        if (sp >= _pods || dp >= _pods) return 0;
        if (st_in_pod >= _tors_per_pod || dt_in_pod >= _tors_per_pod) return 0;
        if (sp == dp) return 0;
        uint32_t cnt = 0;
        for (uint32_t p_src = 0; p_src < _agg_ocs_planes; p_src++) {
            int32_t trunk_i = _tor_to_trunk[sp][p_src][st_in_pod];
            if (trunk_i < 0) continue;
            uint32_t trunk = (uint32_t)trunk_i;
            uint32_t src_gt = sp * _trunk_ports_per_pod + trunk;
            for (uint32_t p_core = 0; p_core < _core_ocs_planes; p_core++) {
                int32_t dst_gt_i = _core_sched.peer[p_core][src_gt];
                if (dst_gt_i < 0) continue;
                uint32_t dst_gt = (uint32_t)dst_gt_i;
                uint32_t peer_pod = dst_gt / _trunk_ports_per_pod;
                uint32_t peer_trunk = dst_gt % _trunk_ports_per_pod;
                if (peer_pod != dp) continue;
                for (uint32_t p_dst = 0; p_dst < _agg_ocs_planes; p_dst++) {
                    int32_t tor2_i = _trunk_to_tor[dp][p_dst][peer_trunk];
                    if (tor2_i < 0) continue;
                    if ((uint32_t)tor2_i != dt_in_pod) continue;
                    cnt++;
                }
            }
        }
        return cnt;
    };

    // Intra-pod matrices.
    for (uint32_t pod = 0; pod < _pods; pod++) {
        out << "### Intra-pod ToR-pair capacity proxy: pod=" << pod << " (ToR_in_pod in [0.." << (_tors_per_pod - 1) << "])\n";
        out << "tor_in_pod";
        for (uint32_t b = 0; b < _tors_per_pod; b++) out << "\t" << b;
        out << "\n";
        for (uint32_t a = 0; a < _tors_per_pod; a++) {
            out << a;
            for (uint32_t b = 0; b < _tors_per_pod; b++) {
                if (a == b) {
                    out << "\t-";
                    continue;
                }
                uint32_t pc = intra_path_count(pod, a, b);
                double bw = pc * link_gbps;
                out << "\t" << pc << "@" << bw;
            }
            out << "\n";
        }
        out << "\n";
    }

    // Cross-pod matrices (by pod-pair).
    for (uint32_t sp = 0; sp < _pods; sp++) {
        for (uint32_t dp = 0; dp < _pods; dp++) {
            if (sp == dp) continue;
            out << "### Cross-pod ToR-pair capacity proxy: src_pod=" << sp << " dst_pod=" << dp << "\n";
            out << "src_tor_in_pod\\dst_tor_in_pod";
            for (uint32_t dt = 0; dt < _tors_per_pod; dt++) out << "\t" << dt;
            out << "\n";
            for (uint32_t st = 0; st < _tors_per_pod; st++) {
                out << st;
                for (uint32_t dt = 0; dt < _tors_per_pod; dt++) {
                    uint32_t pc = cross_path_count(sp, st, dp, dt);
                    double bw = pc * link_gbps;
                    out << "\t" << pc << "@" << bw;
                }
                out << "\n";
            }
            out << "\n";
        }
    }

    out << "## Unreachable GPU-pair examples (up to " << max_unreachable_examples << ")\n";
    uint32_t examples = 0;
    for (uint32_t src = 0; src < _no_of_nodes && examples < max_unreachable_examples; src++) {
        for (uint32_t dst = 0; dst < _no_of_nodes && examples < max_unreachable_examples; dst++) {
            if (src == dst) continue;
            auto* paths = get_bidir_paths(src, dst, false);
            bool ok = (paths && !paths->empty());
            free_paths(paths);
            if (ok) continue;

            uint32_t sp = pod_of_host(src);
            uint32_t dp = pod_of_host(dst);
            uint32_t st = tor_in_pod_of_host(src);
            uint32_t dt = tor_in_pod_of_host(dst);

            out << "- src=" << src << " (pod " << sp << " tor_in_pod " << st << ")"
                << " -> dst=" << dst << " (pod " << dp << " tor_in_pod " << dt << "): ";

            if (sp == dp) {
                out << "UNREACHABLE: same-pod multi-ToR traffic requires Agg-OCS ToR<->ToR direct circuits (current limitation)\n";
                bool has = has_intra_pod_direct_tor_circuit(sp, st, dt, nullptr);
                out << "  detail: direct_tor_circuit=" << (has ? 1 : 0) << "\n";
            } else {
                uint32_t p_src = 0, trunk = 0;
                bool src_has_trunk = has_src_tor_to_any_trunk(sp, st, &p_src, &trunk);
                if (!src_has_trunk) {
                    out << "UNREACHABLE: src ToR has no ToR->trunk mapping in Agg-OCS schedule\n";
                    out << "  detail: src_pod=" << sp << " src_tor_in_pod=" << st << "\n";
                } else {
                    uint32_t p_core = 0, dst_trunk = 0;
                    bool core_hits_dst_pod = has_core_trunk_to_pod(sp, trunk, dp, &p_core, &dst_trunk);
                    if (!core_hits_dst_pod) {
                        out << "UNREACHABLE: Core-OCS does not map src trunk to any trunk in dst pod\n";
                        out << "  detail: src_pod=" << sp << " src_tor_in_pod=" << st
                            << " example(p_src=" << p_src << " trunk=" << trunk << ")"
                            << " dst_pod=" << dp << "\n";
                    } else {
                        uint32_t p_dst = 0;
                        bool dst_has = has_dst_trunk_to_tor(dp, dst_trunk, dt, &p_dst);
                        if (!dst_has) {
                            out << "UNREACHABLE: dst pod has no trunk->ToR mapping for the trunk reached from Core-OCS\n";
                            out << "  detail: src(p_src=" << p_src << " trunk=" << trunk << ")"
                                << " core(p_core=" << p_core << " dst_trunk=" << dst_trunk << ")"
                                << " dst(p_dst=none dst_tor_in_pod=" << dt << ")\n";
                        } else {
                            out << "UNREACHABLE: no (p_src, p_core, p_dst) triple exists under current routing constraints\n";
                            out << "  detail: src_has_trunk=1 core_hits_dst_pod=1 dst_has_trunk_to_tor=1 (but not alignable)\n";
                        }
                    }
                }
            }
            examples++;
        }
    }
}


