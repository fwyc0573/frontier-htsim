// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail3_topology.h"

#include "randomqueue.h"
#include "compositequeue.h"

#include <cassert>
#include <sstream>

static std::string u32s(uint32_t v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

Rail3Topology::Rail3Topology(const Rail3Config& cfg,
                             linkspeed_bps linkspeed,
                             mem_b switch_queuesize_bytes,
                             simtime_picosec link_latency,
                             simtime_picosec switch_latency,
                             queue_type switch_queue_type,
                             queue_type host_queue_type,
                             Logfile* logfile,
                             EventList* eventlist)
    : _pods(cfg.pods)
    , _podsize(cfg.podsize)
    , _gpus_per_server(cfg.gpus_per_server)
    , _servers_per_tor(cfg.servers_per_tor ? cfg.servers_per_tor : cfg.servers_per_pod())
    , _servers_per_pod(cfg.servers_per_pod())
    , _rail_splits_per_pod((_servers_per_tor ? ((_servers_per_pod + _servers_per_tor - 1) / _servers_per_tor) : 1))
    , _tors_per_pod(_gpus_per_server * _rail_splits_per_pod)
    , _tor_up(cfg.tor_up)
    , _agg_count(cfg.agg_count ? cfg.agg_count : cfg.tor_up)
    , _agg_up(cfg.agg_up ? cfg.agg_up : _tors_per_pod)
    , _core_count(_agg_count * _agg_up)
    , _no_of_nodes(_pods * _podsize)
    , _linkspeed(linkspeed)
    , _switch_queuesize_bytes(switch_queuesize_bytes)
    , _link_latency(link_latency)
    , _switch_latency(switch_latency)
    , _switch_queue_type(switch_queue_type)
    , _host_queue_type(host_queue_type)
    , _logfile(logfile)
    , _eventlist(eventlist) {
    assert(_pods > 0);
    assert(_podsize > 0);
    assert(_gpus_per_server > 0);
    assert(_servers_per_tor > 0);
    assert(_tors_per_pod > 0);
    assert(_tor_up > 0);
    assert(_agg_count > 0);
    assert(_agg_up > 0);
    assert(_core_count > 0);
    // Current init_fabric_edges() models a ToR->Agg full mesh of size AggCount.
    // To keep TorUp semantics aligned, we require AggCount == TorUp.
    assert(_agg_count == _tor_up);

    init_host_tor_edges();
    init_fabric_edges();
}

uint32_t Rail3Topology::tor_in_pod_of_host(uint32_t host) const {
    uint32_t h = local_host(host);
    uint32_t rail = rail_of_local_host(h);
    uint32_t srv = server_of_local_host(h);
    uint32_t group = srv / _servers_per_tor;
    if (group >= _rail_splits_per_pod) group = _rail_splits_per_pod - 1;
    return rail * _rail_splits_per_pod + group;
}

HostQueue* Rail3Topology::make_host_queue(const std::string& name) {
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

BaseQueue* Rail3Topology::make_switch_queue(const std::string& name) {
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

Pipe* Rail3Topology::make_pipe(const std::string& name) {
    Pipe* p = new Pipe(_link_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

void Rail3Topology::init_host_tor_edges() {
    _host_to_tor.resize(_no_of_nodes);

    // Small sizes for our use cases; store tor->host for direct indexing.
    _tor_to_host.resize(_pods * _tors_per_pod);
    for (auto& v : _tor_to_host) v.resize(_no_of_nodes);

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

void Rail3Topology::init_fabric_edges() {
    _tor_to_agg.resize(_pods);
    _agg_to_tor.resize(_pods);
    _agg_to_core.resize(_pods);
    _core_to_agg.resize(_pods);

    for (uint32_t pod = 0; pod < _pods; pod++) {
        _tor_to_agg[pod].resize(_tors_per_pod);
        for (uint32_t t = 0; t < _tors_per_pod; t++) _tor_to_agg[pod][t].resize(_agg_count);

        _agg_to_tor[pod].resize(_agg_count);
        for (uint32_t a = 0; a < _agg_count; a++) _agg_to_tor[pod][a].resize(_tors_per_pod);

        _agg_to_core[pod].resize(_agg_count);
        for (uint32_t a = 0; a < _agg_count; a++) _agg_to_core[pod][a].resize(_agg_up);

        _core_to_agg[pod].resize(_core_count);

        // ToR <-> Agg: full mesh inside a pod.
        for (uint32_t t = 0; t < _tors_per_pod; t++) {
            for (uint32_t a = 0; a < _agg_count; a++) {
                _tor_to_agg[pod][t][a].q = make_switch_queue("ToR" + u32s(tor_global(pod, t)) + "->Agg(" + u32s(pod) + "," + u32s(a) + ")");
                _tor_to_agg[pod][t][a].p = make_pipe("P_ToR" + u32s(tor_global(pod, t)) + "_Agg(" + u32s(pod) + "," + u32s(a) + ")");

                _agg_to_tor[pod][a][t].q = make_switch_queue("Agg(" + u32s(pod) + "," + u32s(a) + ")->ToR" + u32s(tor_global(pod, t)));
                _agg_to_tor[pod][a][t].p = make_pipe("P_Agg(" + u32s(pod) + "," + u32s(a) + ")_ToR" + u32s(tor_global(pod, t)));
            }
        }

        // Agg -> Core and Core -> Agg.
        for (uint32_t a = 0; a < _agg_count; a++) {
            for (uint32_t u = 0; u < _agg_up; u++) {
                uint32_t core = a * _agg_up + u;
                _agg_to_core[pod][a][u].q = make_switch_queue("Agg(" + u32s(pod) + "," + u32s(a) + ")->Core" + u32s(core));
                _agg_to_core[pod][a][u].p = make_pipe("P_Agg(" + u32s(pod) + "," + u32s(a) + ")_Core" + u32s(core));

                _core_to_agg[pod][core].q = make_switch_queue("Core" + u32s(core) + "->Agg(" + u32s(pod) + "," + u32s(a) + ")");
                _core_to_agg[pod][core].p = make_pipe("P_Core" + u32s(core) + "_Agg(" + u32s(pod) + "," + u32s(a) + ")");
            }
        }
    }
}

vector<const Route*>* Rail3Topology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    (void)reverse; // symmetric; caller can swap src/dst if needed
    auto* paths = new vector<const Route*>();
    assert(src < _no_of_nodes && dest < _no_of_nodes);
    assert(src != dest);

    uint32_t src_pod = pod_of_host(src);
    uint32_t dst_pod = pod_of_host(dest);
    uint32_t src_tor_in_pod = tor_in_pod_of_host(src);
    uint32_t dst_tor_in_pod = tor_in_pod_of_host(dest);
    uint32_t src_tor = tor_global(src_pod, src_tor_in_pod);
    uint32_t dst_tor = tor_global(dst_pod, dst_tor_in_pod);

    // Same ToR (same pod + same tor): direct.
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

    // Same pod, different ToR: stay within pod (via any Agg).
    if (src_pod == dst_pod) {
        for (uint32_t a = 0; a < _agg_count; a++) {
            auto* r = new Route();
            r->push_back(_host_to_tor[src].q);
            r->push_back(_host_to_tor[src].p);

            if (_switch_latency) {
                auto* sw = new Pipe(_switch_latency, *_eventlist);
                sw->setName("SwLat_ToR" + u32s(src_tor) + "_Agg(" + u32s(src_pod) + "," + u32s(a) + ")");
                r->push_back(sw);
            }

            r->push_back(_tor_to_agg[src_pod][src_tor_in_pod][a].q);
            r->push_back(_tor_to_agg[src_pod][src_tor_in_pod][a].p);

            if (_switch_latency) {
                auto* sw = new Pipe(_switch_latency, *_eventlist);
                sw->setName("SwLat_Agg(" + u32s(src_pod) + "," + u32s(a) + ")");
                r->push_back(sw);
            }

            r->push_back(_agg_to_tor[src_pod][a][dst_tor_in_pod].q);
            r->push_back(_agg_to_tor[src_pod][a][dst_tor_in_pod].p);

            if (_switch_latency) {
                auto* sw = new Pipe(_switch_latency, *_eventlist);
                sw->setName("SwLat_ToR" + u32s(dst_tor) + "_fromAgg(" + u32s(src_pod) + "," + u32s(a) + ")");
                r->push_back(sw);
            }

            r->push_back(_tor_to_host[dst_tor][dest].q);
            r->push_back(_tor_to_host[dst_tor][dest].p);
            paths->push_back(r);
        }
        return paths;
    }

    // Cross pod: choose a core (ECMP across cores).
    for (uint32_t core = 0; core < _core_count; core++) {
        uint32_t a = core / _agg_up;
        uint32_t u = core % _agg_up;
        if (a >= _agg_count) continue;

        auto* r = new Route();
        r->push_back(_host_to_tor[src].q);
        r->push_back(_host_to_tor[src].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(src_tor) + "_Agg(" + u32s(src_pod) + "," + u32s(a) + ")");
            r->push_back(sw);
        }

        // ToR -> Agg(a)
        r->push_back(_tor_to_agg[src_pod][src_tor_in_pod][a].q);
        r->push_back(_tor_to_agg[src_pod][src_tor_in_pod][a].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_Agg(" + u32s(src_pod) + "," + u32s(a) + ")_Core" + u32s(core));
            r->push_back(sw);
        }

        // Agg(a) -> Core(core)
        r->push_back(_agg_to_core[src_pod][a][u].q);
        r->push_back(_agg_to_core[src_pod][a][u].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_Core" + u32s(core));
            r->push_back(sw);
        }

        // Core(core) -> Agg(a) in dst pod
        r->push_back(_core_to_agg[dst_pod][core].q);
        r->push_back(_core_to_agg[dst_pod][core].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_Agg(" + u32s(dst_pod) + "," + u32s(a) + ")_fromCore" + u32s(core));
            r->push_back(sw);
        }

        // Agg(a) -> dst ToR
        r->push_back(_agg_to_tor[dst_pod][a][dst_tor_in_pod].q);
        r->push_back(_agg_to_tor[dst_pod][a][dst_tor_in_pod].p);

        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_ToR" + u32s(dst_tor) + "_fromAgg(" + u32s(dst_pod) + "," + u32s(a) + ")");
            r->push_back(sw);
        }

        r->push_back(_tor_to_host[dst_tor][dest].q);
        r->push_back(_tor_to_host[dst_tor][dest].p);
        paths->push_back(r);
    }

    return paths;
}

vector<uint32_t>* Rail3Topology::get_neighbours(uint32_t src) {
    // Same heuristic as RailTopology: neighbours are GPUs on the same rail index within a pod.
    auto* n = new vector<uint32_t>();
    if (src >= _no_of_nodes) return n;
    uint32_t pod = pod_of_host(src);
    uint32_t h = local_host(src);
    uint32_t rail = rail_of_local_host(h);
    for (uint32_t i = 0; i < _podsize; i++) {
        uint32_t other = pod * _podsize + i;
        if (other == src) continue;
        if (rail_of_local_host(i) == rail) n->push_back(other);
    }
    return n;
}


