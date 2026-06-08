// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_ocs_topology.h"

#include "randomqueue.h"
#include "queue.h"
#include "compositequeue.h"

#include <cassert>
#include <sstream>

static std::string u32s(uint32_t v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

// A FIFO that never drops packets due to buffer overflow.
// This approximates an "ideal circuit" where loss is not modeled at the fabric.
class NoDropQueue final : public Queue {
public:
    NoDropQueue(linkspeed_bps bitrate, mem_b maxsize, EventList& eventlist, QueueLogger* logger)
        : Queue(bitrate, maxsize, eventlist, logger) {}

    void receivePacket(Packet& pkt) override {
        // Identical to Queue::receivePacket but without the overflow drop check.
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

RailOcsTopology::RailOcsTopology(uint32_t servers,
                                 uint32_t gpus_per_server,
                                 uint32_t servers_per_tor,
                                 uint32_t planes,
                                 const RailOcsStaticSchedule& schedule,
                                 linkspeed_bps linkspeed,
                                 mem_b switch_queuesize_bytes,
                                 mem_b ocs_queuesize_bytes,
                                 simtime_picosec link_latency,
                                 simtime_picosec switch_latency,
                                 simtime_picosec ocs_link_latency,
                                 simtime_picosec ocs_switch_latency,
                                 bool ocs_no_drop,
                                 const std::string& ocs_queue_type,
                                 bool ocs_cut_through,
                                 queue_type switch_queue_type,
                                 queue_type host_queue_type,
                                 Logfile* logfile,
                                 EventList* eventlist)
    : _servers(servers),
      _gpus_per_server(gpus_per_server),
      _servers_per_tor(servers_per_tor ? servers_per_tor : servers),
      _rail_splits((servers + (_servers_per_tor - 1)) / _servers_per_tor),
      _tors(gpus_per_server * _rail_splits),
      _planes(planes),
      _no_of_nodes(servers * gpus_per_server),
      _schedule(schedule),
      _linkspeed(linkspeed),
      _switch_queuesize_bytes(switch_queuesize_bytes),
      _ocs_queuesize_bytes(ocs_queuesize_bytes),
      _link_latency(link_latency),
      _switch_latency(switch_latency),
      _ocs_link_latency(ocs_link_latency),
      _ocs_switch_latency(ocs_switch_latency),
      _ocs_no_drop(ocs_no_drop),
      _ocs_queue_type(ocs_queue_type),
      _ocs_cut_through(ocs_cut_through),
      _switch_queue_type(switch_queue_type),
      _host_queue_type(host_queue_type),
      _logfile(logfile),
      _eventlist(eventlist) {
    assert(_servers > 0);
    assert(_gpus_per_server > 0);
    assert(_servers_per_tor > 0);
    assert(_rail_splits > 0);
    assert(_planes > 0);
    assert(_schedule.planes == _planes);
    assert(_schedule.tors == _tors);
    init_network();
}

uint32_t RailOcsTopology::tor_of_host(uint32_t host) const {
    // Split each rail into multiple ToRs if servers_per_tor < servers.
    // Each rail has _rail_splits ToRs, indexed by group = server_id / servers_per_tor.
    uint32_t rail = rail_of_host(host);
    uint32_t srv = server_of_host(host);
    uint32_t group = srv / _servers_per_tor;
    if (group >= _rail_splits) group = _rail_splits - 1;
    return rail * _rail_splits + group;
}

HostQueue* RailOcsTopology::make_host_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    HostQueue* q = nullptr;
    if (_host_queue_type == PRIORITY) {
        q = new PriorityQueue(_linkspeed, _switch_queuesize_bytes, *_eventlist, ql);
    } else {
        q = new FairPriorityQueue(_linkspeed, _switch_queuesize_bytes, *_eventlist, ql);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* RailOcsTopology::make_switch_queue(const std::string& name) {
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

BaseQueue* RailOcsTopology::make_ocs_queue(const std::string& name) {
    // OCS circuits are not packet switches; we model them as a FIFO serializer.
    // If _ocs_no_drop is set, do not model loss in the fabric (never drop on overflow).
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
        if (_ocs_queue_type.empty() || _ocs_queue_type == "fifo") {
            q = new Queue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
        } else if (_ocs_queue_type == "composite") {
            q = new CompositeQueue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
        } else if (_ocs_queue_type == "random") {
            q = new RandomQueue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql, 0);
        } else {
            // Defensive fallback: config loader should have validated.
            q = new Queue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
        }
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* RailOcsTopology::maybe_make_ocs_queue(const std::string& name, bool tor_egress) {
    if (_ocs_cut_through) {
        // Cut-through modeling (bandwidth-preserving):
        // - Keep serialization/queueing at the transmitter (ToR egress).
        // - Remove internal OCS queues to avoid adding store-and-forward hops.
        if (!tor_egress) {
            (void)name;
            return nullptr;
        }
        return make_ocs_queue(name);
    }
    return make_ocs_queue(name);
}

Pipe* RailOcsTopology::make_pipe(const std::string& name) {
    auto* p = new Pipe(_link_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

Pipe* RailOcsTopology::make_ocs_pipe(const std::string& name) {
    auto* p = new Pipe(_ocs_link_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

void RailOcsTopology::init_network() {
    _host_to_tor.resize(_no_of_nodes);
    _tor_to_host.assign(_tors, std::vector<Edge>(_no_of_nodes));
    _tor_to_tor.assign(_planes,
                       std::vector<std::vector<Edge>>(_tors, std::vector<Edge>(_tors)));
    _tor_ocs_egress.assign(_planes, std::vector<Edge>(_tors));

    // Host <-> ToR
    for (uint32_t h = 0; h < _no_of_nodes; h++) {
        uint32_t tor = tor_of_host(h);
        _host_to_tor[h].q = make_host_queue("H" + u32s(h) + "->ToR" + u32s(tor));
        _host_to_tor[h].p = make_pipe("P_H" + u32s(h) + "_ToR" + u32s(tor));

        _tor_to_host[tor][h].q = make_switch_queue("ToR" + u32s(tor) + "->H" + u32s(h));
        _tor_to_host[tor][h].p = make_pipe("P_ToR" + u32s(tor) + "_H" + u32s(h));
    }

    // OCS ToR <-> ToR links according to static schedule (slot 0)
    for (uint32_t p = 0; p < _planes; p++) {
        for (uint32_t a = 0; a < _tors; a++) {
            int32_t b = _schedule.peer[p][a];
            if (b < 0) continue;
            // Create directed edges for both directions only once
            if (a < (uint32_t)b) {
                uint32_t bb = (uint32_t)b;
                // Cut-through mode:
                // - keep one serializer per (plane,src_tor), and make internal OCS edge Pipe-only.
                // Non cut-through (legacy):
                // - keep per-edge queue+pipe.
                if (_ocs_cut_through) {
                    if (!_tor_ocs_egress[p][a].q) _tor_ocs_egress[p][a].q = make_ocs_queue("OCS_EGRESS" + u32s(p) + " ToR" + u32s(a));
                    if (!_tor_ocs_egress[p][bb].q) _tor_ocs_egress[p][bb].q = make_ocs_queue("OCS_EGRESS" + u32s(p) + " ToR" + u32s(bb));
                } else {
                    _tor_to_tor[p][a][bb].q = make_ocs_queue("OCS" + u32s(p) + " ToR" + u32s(a) + "->ToR" + u32s(bb));
                }
                _tor_to_tor[p][a][bb].p = make_ocs_pipe("P_OCS" + u32s(p) + "_ToR" + u32s(a) + "_ToR" + u32s(bb));

                if (!_ocs_cut_through) {
                    _tor_to_tor[p][bb][a].q = make_ocs_queue("OCS" + u32s(p) + " ToR" + u32s(bb) + "->ToR" + u32s(a));
                }
                _tor_to_tor[p][bb][a].p = make_ocs_pipe("P_OCS" + u32s(p) + "_ToR" + u32s(bb) + "_ToR" + u32s(a));
            }
        }
    }
}

vector<const Route*>* RailOcsTopology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    (void)reverse;
    auto* paths = new vector<const Route*>();
    assert(src < _no_of_nodes && dest < _no_of_nodes);
    assert(src != dest);

    const uint32_t src_tor = tor_of_host(src);
    const uint32_t dst_tor = tor_of_host(dest);

    // Same ToR: direct within ToR
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

    // Different ToR: only if connected by some OCS plane (static).
    for (uint32_t p = 0; p < _planes; p++) {
        auto& e = _tor_to_tor[p][src_tor][dst_tor];
        if (!e.p) continue;
        if (!_ocs_cut_through && !e.q) continue;

        auto* r = new Route();
        r->push_back(_host_to_tor[src].q);
        r->push_back(_host_to_tor[src].p);

        if (_ocs_switch_latency) {
            auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(src_tor) + "_OCS" + u32s(p));
            r->push_back(sw);
        }

        // In cut-through mode, the internal OCS edge has no queue; use ToR egress serializer instead.
        if (_ocs_cut_through) {
            auto& eg = _tor_ocs_egress[p][src_tor];
            if (eg.q) r->push_back(eg.q);
        } else {
            r->push_back(e.q);
        }
        r->push_back(e.p);

        if (_ocs_switch_latency) {
            auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(dst_tor) + "_OCS" + u32s(p));
            r->push_back(sw);
        }

        r->push_back(_tor_to_host[dst_tor][dest].q);
        r->push_back(_tor_to_host[dst_tor][dest].p);
        paths->push_back(r);
    }

    return paths;
}

vector<uint32_t>* RailOcsTopology::get_neighbours(uint32_t src) {
    auto* n = new vector<uint32_t>();
    if (src >= _no_of_nodes) return n;
    const uint32_t rail = rail_of_host(src);
    for (uint32_t h = 0; h < _no_of_nodes; h++) {
        if (h == src) continue;
        if (rail_of_host(h) == rail) n->push_back(h);
    }
    return n;
}


