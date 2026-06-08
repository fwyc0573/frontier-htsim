// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_topology.h"

#include "main.h"
#include "loggers.h"
#include "queue.h"
#include "compositequeue.h"

#include <cassert>
#include <sstream>

static std::string u32s(uint32_t v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

RailTopology::RailTopology(uint32_t servers,
                           uint32_t gpus_per_server,
                           uint32_t spines,
                           linkspeed_bps linkspeed,
                           mem_b queuesize_bytes,
                           simtime_picosec link_latency,
                           simtime_picosec switch_latency,
                           queue_type switch_queue_type,
                           queue_type host_queue_type,
                           uint32_t servers_per_tor,
                           Logfile* logfile,
                           EventList* eventlist)
    : _servers(servers),
      _gpus_per_server(gpus_per_server),
      _servers_per_tor(servers_per_tor ? servers_per_tor : servers),
      _rail_splits((servers + (_servers_per_tor - 1)) / _servers_per_tor),
      _tors(gpus_per_server * _rail_splits),
      _spines(spines),
      _no_of_nodes(servers * gpus_per_server),
      _linkspeed(linkspeed),
      _queuesize_bytes(queuesize_bytes),
      _link_latency(link_latency),
      _switch_latency(switch_latency),
      _switch_queue_type(switch_queue_type),
      _host_queue_type(host_queue_type),
      _logfile(logfile),
      _eventlist(eventlist) {
    assert(_servers > 0);
    assert(_gpus_per_server > 0);
    assert(_spines > 0);
    assert(_servers_per_tor > 0);
    assert(_rail_splits > 0);
    init_network();
}

uint32_t RailTopology::tor_of_host(uint32_t host) const {
    // Split each rail into multiple ToRs if servers_per_tor < servers.
    // Each rail has _rail_splits ToRs, indexed by group = server_id / servers_per_tor.
    uint32_t rail = rail_of_host(host);
    uint32_t srv = server_of_host(host);
    uint32_t group = srv / _servers_per_tor;
    if (group >= _rail_splits) group = _rail_splits - 1; // last bucket for any remainder
    return rail * _rail_splits + group;
}

BaseQueue* RailTopology::make_switch_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    BaseQueue* q = nullptr;
    // Align with FatTreeTopology's default (COMPOSITE) when requested.
    if (_switch_queue_type == COMPOSITE || _switch_queue_type == COMPOSITE_ECN || _switch_queue_type == COMPOSITE_ECN_LB) {
        q = new CompositeQueue(_linkspeed, _queuesize_bytes, *_eventlist, ql);
    } else {
        // Fallback: keep legacy behavior.
        q = new RandomQueue(_linkspeed, _queuesize_bytes, *_eventlist, ql, 0);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

HostQueue* RailTopology::make_host_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }

    HostQueue* q = nullptr;
    if (_host_queue_type == PRIORITY) {
        q = new PriorityQueue(_linkspeed, _queuesize_bytes, *_eventlist, ql);
    } else {
        // default to fair priority
        q = new FairPriorityQueue(_linkspeed, _queuesize_bytes, *_eventlist, ql);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

Pipe* RailTopology::make_pipe(const std::string& name) {
    auto* p = new Pipe(_link_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

void RailTopology::init_network() {
    _host_to_tor.resize(_no_of_nodes);
    _tor_to_host.assign(_tors, std::vector<Edge>(_no_of_nodes));

    _tor_to_spine.assign(_tors, std::vector<Edge>(_spines));
    _spine_to_tor.assign(_spines, std::vector<Edge>(_tors));

    // Host <-> ToR (rail) links
    for (uint32_t h = 0; h < _no_of_nodes; h++) {
        uint32_t tor = tor_of_host(h);

        // Host egress must be a HostQueue for NDP (see NdpSrc::send_packet assert).
        _host_to_tor[h].q = make_host_queue("H" + u32s(h) + "->ToR" + u32s(tor));
        _host_to_tor[h].p = make_pipe("P_H" + u32s(h) + "_ToR" + u32s(tor));

        _tor_to_host[tor][h].q = (BaseQueue*)make_switch_queue("ToR" + u32s(tor) + "->H" + u32s(h));
        _tor_to_host[tor][h].p = make_pipe("P_ToR" + u32s(tor) + "_H" + u32s(h));
    }

    // ToR <-> Spine full mesh
    for (uint32_t tor = 0; tor < _tors; tor++) {
        for (uint32_t s = 0; s < _spines; s++) {
            _tor_to_spine[tor][s].q = make_switch_queue("ToR" + u32s(tor) + "->Sp" + u32s(s));
            _tor_to_spine[tor][s].p = make_pipe("P_ToR" + u32s(tor) + "_Sp" + u32s(s));

            _spine_to_tor[s][tor].q = make_switch_queue("Sp" + u32s(s) + "->ToR" + u32s(tor));
            _spine_to_tor[s][tor].p = make_pipe("P_Sp" + u32s(s) + "_ToR" + u32s(tor));
        }
    }
}

vector<const Route*>* RailTopology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    (void)reverse; // We construct symmetric forward/reverse; caller can request reverse routes by swapping endpoints.
    auto* paths = new vector<const Route*>();
    assert(src < _no_of_nodes && dest < _no_of_nodes);
    assert(src != dest);

    const uint32_t src_tor = tor_of_host(src);
    const uint32_t dst_tor = tor_of_host(dest);

    // If src and dst are on the same rail/ToR: single path (ToR acts as L2 switch).
    if (src_tor == dst_tor) {
        auto* r = new Route();
        r->push_back(_host_to_tor[src].q);
        r->push_back(_host_to_tor[src].p);

        // optional switch latency at ToR
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

    // Different rails: one path per spine (ECMP across spines).
    for (uint32_t s = 0; s < _spines; s++) {
        auto* r = new Route();

        // src host -> src ToR
        r->push_back(_host_to_tor[src].q);
        r->push_back(_host_to_tor[src].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(src_tor) + "_Sp" + u32s(s));
            r->push_back(sw);
        }

        // src ToR -> spine
        r->push_back(_tor_to_spine[src_tor][s].q);
        r->push_back(_tor_to_spine[src_tor][s].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_Sp" + u32s(s));
            r->push_back(sw);
        }

        // spine -> dst ToR
        r->push_back(_spine_to_tor[s][dst_tor].q);
        r->push_back(_spine_to_tor[s][dst_tor].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(dst_tor) + "_fromSp" + u32s(s));
            r->push_back(sw);
        }

        // dst ToR -> dst host
        r->push_back(_tor_to_host[dst_tor][dest].q);
        r->push_back(_tor_to_host[dst_tor][dest].p);

        paths->push_back(r);
    }
    return paths;
}

vector<uint32_t>* RailTopology::get_neighbours(uint32_t src) {
    // "Neighbours" in this simulator context is usually used for local traffic heuristics.
    // For a rail topology, define neighbours as other GPUs on the same rail (same ToR).
    auto* n = new vector<uint32_t>();
    if (src >= _no_of_nodes) return n;

    const uint32_t rail = rail_of_host(src);
    for (uint32_t h = 0; h < _no_of_nodes; h++) {
        if (h == src) continue;
        if (rail_of_host(h) == rail) n->push_back(h);
    }
    return n;
}

void RailTopology::add_switch_loggers(Logfile& log, simtime_picosec sample_period) {
    // Best-effort: add empty loggers for ToR->spine and spine->ToR queues.
    // Host->ToR queues are usually not counted as "switch queues" here.
    for (uint32_t tor = 0; tor < _tors; tor++) {
        for (uint32_t s = 0; s < _spines; s++) {
            if (_tor_to_spine[tor][s].q) {
                auto* l = new QueueLoggerSampling(sample_period, *_eventlist);
                log.addLogger(*l);
                // RandomQueue doesn't expose a public setter for logger in this codebase.
                // We still register the logger; queue-level attachment is best-effort and may be ignored.
            }
            if (_spine_to_tor[s][tor].q) {
                auto* l = new QueueLoggerSampling(sample_period, *_eventlist);
                log.addLogger(*l);
                // See comment above.
            }
        }
    }
}


