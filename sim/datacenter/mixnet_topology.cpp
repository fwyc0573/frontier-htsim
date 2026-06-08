// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "mixnet_topology.h"

#include "pipe.h"

#include <algorithm>
#include <cassert>
#include <sstream>
#include "route.h"

static std::string u32s(uint32_t v) {
    std::stringstream ss;
    ss << v;
    return ss.str();
}

static void free_paths(vector<const Route*>* paths) {
    if (!paths) return;
    for (auto* r : *paths) {
        delete r;
    }
    delete paths;
}

MixNetTopology::MixNetTopology(Topology* eps,
                               Topology* ocs,
                               std::vector<uint32_t> ep_group_id,
                               uint32_t gpus_per_server,
                               std::vector<uint32_t> eps_gateway_local_indices,
                               linkspeed_bps intra_linkspeed,
                               mem_b intra_queue_bytes,
                               simtime_picosec intra_latency,
                               queue_type host_queue_type,
                               Logfile* logfile,
                               EventList* eventlist)
    : _eps(eps),
      _ocs(ocs),
      _ep_group_id(std::move(ep_group_id)),
      _no_of_nodes(0),
      _gpus_per_server(gpus_per_server),
      _eps_gateway_local_indices(std::move(eps_gateway_local_indices)),
      _intra_linkspeed(intra_linkspeed),
      _intra_queue_bytes(intra_queue_bytes),
      _intra_latency(intra_latency),
      _host_queue_type(host_queue_type),
      _logfile(logfile),
      _eventlist(eventlist) {
    assert(_eps != nullptr);
    assert(_ocs != nullptr);
    assert(_eventlist != nullptr);
    assert(_gpus_per_server > 0);
    assert(!_eps_gateway_local_indices.empty());
    _no_of_nodes = (uint32_t)_ep_group_id.size();

    const uint32_t k = (uint32_t)_eps_gateway_local_indices.size();
    _to_gw.assign(_no_of_nodes, std::vector<Edge>(k));
    _from_gw.assign(_no_of_nodes, std::vector<Edge>(k));

    for (uint32_t n = 0; n < _no_of_nodes; n++) {
        for (uint32_t gi = 0; gi < k; gi++) {
            const uint32_t gw = gw_node(n, gi);
            if (gw >= _no_of_nodes) {
                assert(false && "MixNetTopology: gw node out of range");
            }
            if (gw == n) continue;
            _to_gw[n][gi].q = make_intra_host_queue("Intra H" + u32s(n) + "->GW" + u32s(gw));
            _to_gw[n][gi].p = make_intra_pipe("P_Intra_" + u32s(n) + "_GW" + u32s(gw));
            _from_gw[n][gi].q = make_intra_host_queue("Intra GW" + u32s(gw) + "->H" + u32s(n));
            _from_gw[n][gi].p = make_intra_pipe("P_Intra_GW" + u32s(gw) + "_" + u32s(n));
        }
    }

    // Build mapping: gateway full node_id -> gw-only EPS node_id [0..servers*k-1]
    assert(_no_of_nodes % _gpus_per_server == 0);
    const uint32_t servers = _no_of_nodes / _gpus_per_server;
    _full_to_eps.assign(_no_of_nodes, -1);
    for (uint32_t s = 0; s < servers; s++) {
        for (uint32_t gi = 0; gi < k; gi++) {
            uint32_t full_id = s * _gpus_per_server + _eps_gateway_local_indices[gi];
            uint32_t eps_id = s * k + gi;
            if (full_id < _no_of_nodes) _full_to_eps[full_id] = (int32_t)eps_id;
        }
    }
}

uint32_t MixNetTopology::gw_node(uint32_t node, uint32_t k) const {
    const uint32_t srv = server_of(node);
    const uint32_t idx = _eps_gateway_local_indices.at(k);
    return srv * _gpus_per_server + idx;
}

int32_t MixNetTopology::gw_eps_id(uint32_t full_gw_node) const {
    if (full_gw_node >= _no_of_nodes) return -1;
    return _full_to_eps[full_gw_node];
}

HostQueue* MixNetTopology::make_intra_host_queue(const std::string& name) {
    QueueLoggerSampling* ql = nullptr;
    if (_logfile) {
        ql = new QueueLoggerSampling(timeFromMs(1000), *_eventlist);
        _logfile->addLogger(*ql);
    }
    HostQueue* q = nullptr;
    if (_host_queue_type == PRIORITY) {
        q = new PriorityQueue(_intra_linkspeed, _intra_queue_bytes, *_eventlist, ql);
    } else {
        q = new FairPriorityQueue(_intra_linkspeed, _intra_queue_bytes, *_eventlist, ql);
    }
    q->setName(name);
    if (_logfile) _logfile->writeName(*q);
    return q;
}

Pipe* MixNetTopology::make_intra_pipe(const std::string& name) {
    auto* p = new Pipe(_intra_latency, *_eventlist);
    p->setName(name);
    if (_logfile) _logfile->writeName(*p);
    return p;
}

vector<const Route*>* MixNetTopology::get_bidir_paths(uint32_t src, uint32_t dest, bool reverse) {
    if (src >= _no_of_nodes || dest >= _no_of_nodes) {
        // Keep behavior consistent with other topologies: invalid node ids should crash early.
        assert(false && "MixNetTopology: src/dest out of range");
    }
    bool same_group = (_ep_group_id[src] == _ep_group_id[dest]);
    if (same_group) {
        auto* ocs_paths = _ocs->get_bidir_paths(src, dest, reverse);
        if (ocs_paths && !ocs_paths->empty()) {
            return ocs_paths;
        }
        // Fallback semantics: if OCS is unreachable under static slot0 matching, route via EPS gateways.
        free_paths(ocs_paths);
    }

    const uint32_t gw_k = (uint32_t)(_ep_group_id[dest] % _eps_gateway_local_indices.size());
    const uint32_t gw_src = gw_node(src, gw_k);
    const uint32_t gw_dst = gw_node(dest, gw_k);

    const int32_t eps_src = gw_eps_id(gw_src);
    const int32_t eps_dst = gw_eps_id(gw_dst);
    if (eps_src < 0 || eps_dst < 0) {
        // Misconfig: gateway nodes must be part of the gw-only EPS topology.
        return new vector<const Route*>();
    }

    auto* out = new vector<const Route*>();

    // Special case: same EPS gateway node (e.g., cross-group but same server + same gw choice).
    if (eps_src == eps_dst) {
        auto* r = new Route();
        if (src != gw_src) {
            auto& e = _to_gw[src][gw_k];
            if (!e.q || !e.p) return out;
            r->push_back(e.q);
            r->push_back(e.p);
        }
        if (dest != gw_dst) {
            auto& e = _from_gw[dest][gw_k];
            if (!e.q || !e.p) return out;
            r->push_back(e.q);
            r->push_back(e.p);
        }
        out->push_back(r);
        return out;
    }

    // EPS segment runs between gateways (gw-only node ids).
    auto* eps_paths = _eps->get_bidir_paths((uint32_t)eps_src, (uint32_t)eps_dst, reverse);
    if (!eps_paths) return eps_paths;
    for (auto* base : *eps_paths) {
        if (!base) continue;
        auto* r = new Route();

        // src -> gw_src (intra, cost ~0)
        if (src != gw_src) {
            auto& e = _to_gw[src][gw_k];
            if (!e.q || !e.p) continue;
            r->push_back(e.q);
            r->push_back(e.p);
        }

        // gw_src -> gw_dst (EPS fabric; gw-only ids)
        for (size_t i = 0; i < base->size(); i++) {
            r->push_back(base->at(i));
        }

        // gw_dst -> dest (intra, cost ~0)
        if (dest != gw_dst) {
            auto& e = _from_gw[dest][gw_k];
            if (!e.q || !e.p) continue;
            r->push_back(e.q);
            r->push_back(e.p);
        }

        out->push_back(r);
    }
    return out;
}

vector<uint32_t>* MixNetTopology::get_neighbours(uint32_t src) {
    // Conservative + safe:
    // - If src is an EPS gateway node, return neighbours from the gw-only EPS topology (by mapped eps_id).
    // - Otherwise, return empty (we don't enumerate "implicit intra-server" neighbours here).
    auto* n = new vector<uint32_t>();
    if (src >= _no_of_nodes) return n;
    int32_t eps_id = gw_eps_id(src);
    if (eps_id >= 0) {
        delete n;
        return _eps->get_neighbours((uint32_t)eps_id);
    }
    return n;
}

