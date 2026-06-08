// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#include "rail_delay_switch.h"

#include "callback_pipe.h"

RailDelaySwitch::RailDelaySwitch(EventList& eventlist, const std::string& name, simtime_picosec delay)
    : Switch(eventlist, name) {
    _pipe = new CallbackPipe(delay, eventlist, this);
}

RailDelaySwitch::~RailDelaySwitch() {
    delete _pipe;
    _pipe = nullptr;
}

void RailDelaySwitch::receivePacket(Packet& pkt) {
    // Two-phase processing, same pattern as FatTreeSwitch:
    // - First arrival: enqueue in internal pipeline (CallbackPipe) to model switch latency.
    // - Callback: pop and forward to next hop in the already-fixed end-to-end route.
    Packet* p = &pkt;
    auto it = _inflight.find(p);
    if (it == _inflight.end()) {
        _inflight.insert(p);
        _pipe->receivePacket(pkt);
        return;
    }

    _inflight.erase(it);
    pkt.sendOn();
}


