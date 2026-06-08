// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "ep_group_ocs_topology.h"

#include "queue.h"
#include "compositequeue.h"
#include "randomqueue.h"

#include <cassert>
#include <fstream>
#include <sstream>

static std::string u32s(uint32_t v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

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
        if (queueWasEmpty) beginService();
    }
};

EpGroupOcsTopology::EpGroupOcsTopology(const std::vector<uint32_t>& group_id,
                                     const std::vector<uint32_t>& local_port,
                                     uint32_t group_size,
                                     const RailOcsStaticSchedule& schedule,
                                     linkspeed_bps linkspeed,
                                     mem_b host_queuesize_bytes,
                                     mem_b ocs_queuesize_bytes,
                                     simtime_picosec link_latency,
                                     simtime_picosec switch_latency,
                                     simtime_picosec ocs_link_latency,
                                     simtime_picosec ocs_switch_latency,
                                     bool ocs_no_drop,
                                     queue_type host_queue_type,
                                     Logfile* logfile,
                                     EventList* eventlist)
    : _no_of_nodes((uint32_t)group_id.size()),
      _group_size(group_size),
      _planes(schedule.planes),
      _group_id(group_id),
      _local_port(local_port),
      _schedule(schedule),
      _linkspeed(linkspeed),
      _host_queuesize_bytes(host_queuesize_bytes),
      _ocs_queuesize_bytes(ocs_queuesize_bytes),
      _link_latency(link_latency),
      _switch_latency(switch_latency),
      _ocs_link_latency(ocs_link_latency),
      _ocs_switch_latency(ocs_switch_latency),
      _ocs_no_drop(ocs_no_drop),
      _host_queue_type(host_queue_type),
      _logfile(logfile),
      _eventlist(eventlist) {
    assert(_no_of_nodes > 0);
    assert(_local_port.size() == _no_of_nodes);
    assert(_group_size > 0);
    assert(_schedule.tors == _group_size);
    assert(_planes > 0);

    _host_q.assign(_no_of_nodes, nullptr);
    _host_in.assign(_no_of_nodes, Edge());
    _ocs_out.assign(_planes, std::vector<Edge>(_no_of_nodes));

    // Build inverse mapping: (group_id, local_port) -> node
    uint32_t max_gid = 0;
    for (auto g : _group_id) if (g > max_gid) max_gid = g;
    const uint32_t groups = max_gid + 1;
    std::vector<std::vector<uint32_t>> group_nodes(groups, std::vector<uint32_t>(_group_size, (uint32_t)-1));
    for (uint32_t n = 0; n < _no_of_nodes; n++) {
        uint32_t g = _group_id[n];
        uint32_t lp = _local_port[n];
        assert(g < groups);
        assert(lp < _group_size);
        group_nodes[g][lp] = n;
    }
    // Create host queues and receiver NIC queues/pipes.
    for (uint32_t n = 0; n < _no_of_nodes; n++) {
        _host_q[n] = make_host_queue("H" + u32s(n) + "->EPG_OCS");
        _host_in[n].q = make_host_in_queue("EPG_OCS->H" + u32s(n));
        _host_in[n].p = make_pipe("P_EPG_OCS_H" + u32s(n), _link_latency);
    }

    // Create OCS circuit edges per group per plane according to schedule.
    for (uint32_t g = 0; g < groups; g++) {
        for (uint32_t p = 0; p < _planes; p++) {
            for (uint32_t a = 0; a < _group_size; a++) {
                int32_t b = _schedule.peer[p][a];
                if (b < 0) continue;
                uint32_t bb = (uint32_t)b;
                if (a < bb) {
                    uint32_t na = group_nodes[g][a];
                    uint32_t nb = group_nodes[g][bb];
                    if (na == (uint32_t)-1 || nb == (uint32_t)-1) continue; // allow sparse groups
                    _ocs_out[p][na].q = make_ocs_queue("OCS" + u32s(p) + " G" + u32s(g) + " " + u32s(a) + "->" + u32s(bb));
                    _ocs_out[p][na].p = make_pipe("P_OCS" + u32s(p) + "_G" + u32s(g) + "_" + u32s(a) + "_" + u32s(bb), _ocs_link_latency);

                    _ocs_out[p][nb].q = make_ocs_queue("OCS" + u32s(p) + " G" + u32s(g) + " " + u32s(bb) + "->" + u32s(a));
                    _ocs_out[p][nb].p = make_pipe("P_OCS" + u32s(p) + "_G" + u32s(g) + "_" + u32s(bb) + "_" + u32s(a), _ocs_link_latency);
                }
            }
        }
    }
}

HostQueue* EpGroupOcsTopology::make_host_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    HostQueue* q = nullptr;
    if (_host_queue_type == PRIORITY) {
        q = new PriorityQueue(_linkspeed, _host_queuesize_bytes, *_eventlist, ql);
    } else {
        q = new FairPriorityQueue(_linkspeed, _host_queuesize_bytes, *_eventlist, ql);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* EpGroupOcsTopology::make_host_in_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    // Receiver-side serializer (simple FIFO).
    Queue* q = new Queue(_linkspeed, _host_queuesize_bytes, *_eventlist, ql);
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

BaseQueue* EpGroupOcsTopology::make_ocs_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    Queue* q = nullptr;
    if (_ocs_no_drop) {
        q = new NoDropQueue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
    } else {
        q = new Queue(_linkspeed, _ocs_queuesize_bytes, *_eventlist, ql);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

Pipe* EpGroupOcsTopology::make_pipe(const std::string& name, simtime_picosec lat) {
    auto* p = new Pipe(lat, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

vector<const Route*>* EpGroupOcsTopology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    (void)reverse;
    auto* paths = new vector<const Route*>();
    assert(src < _no_of_nodes && dest < _no_of_nodes);
    assert(src != dest);
    if (_group_id[src] != _group_id[dest]) {
        return paths; // unreachable across EP groups in OCS fabric
    }
    uint32_t lp_src = _local_port[src];
    uint32_t lp_dst = _local_port[dest];
    if (lp_src >= _group_size || lp_dst >= _group_size) return paths;

    for (uint32_t p = 0; p < _planes; p++) {
        if (_schedule.peer[p][lp_src] != (int32_t)lp_dst) continue;
        auto& e = _ocs_out[p][src];
        if (!e.q || !e.p) continue;
        auto* r = new Route();
        r->push_back(_host_q[src]);
        if (_switch_latency) {
            auto* sw = new Pipe(_switch_latency, *_eventlist);
            sw->setName("SwLat_H" + u32s(src));
            r->push_back(sw);
        }
        if (_ocs_switch_latency) {
            auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
            sw->setName("SwLat_OCS" + u32s(p) + "_H" + u32s(src));
            r->push_back(sw);
        }
        r->push_back(e.q);
        r->push_back(e.p);
        if (_ocs_switch_latency) {
            auto* sw = new Pipe(_ocs_switch_latency, *_eventlist);
            sw->setName("SwLat_OCS" + u32s(p) + "_H" + u32s(dest));
            r->push_back(sw);
        }
        r->push_back(_host_in[dest].q);
        r->push_back(_host_in[dest].p);
        paths->push_back(r);
    }
    return paths;
}

vector<uint32_t>* EpGroupOcsTopology::get_neighbours(uint32_t src) {
    auto* n = new vector<uint32_t>();
    if (src >= _no_of_nodes) return n;
    uint32_t g = _group_id[src];
    for (uint32_t i = 0; i < _no_of_nodes; i++) {
        if (i == src) continue;
        if (_group_id[i] == g) n->push_back(i);
    }
    return n;
}

