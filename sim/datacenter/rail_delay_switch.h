// -*- c-basic-offset: 4; indent-tabs-mode: nil -*-
#ifndef RAIL_DELAY_SWITCH_H
#define RAIL_DELAY_SWITCH_H

#include "switch.h"

#include <unordered_set>

class CallbackPipe;

// A minimal "switch node" that models an ingress->egress switching delay (switch_latency)
// but does not perform routing. This is useful when routes are precomputed end-to-end,
// yet we still want to model a per-hop switch pipeline similar to FatTreeSwitch.
class RailDelaySwitch final : public Switch {
public:
    RailDelaySwitch(EventList& eventlist, const std::string& name, simtime_picosec delay);
    ~RailDelaySwitch() override;

    void receivePacket(Packet& pkt) override;

private:
    CallbackPipe* _pipe;
    std::unordered_set<Packet*> _inflight;
};

#endif // RAIL_DELAY_SWITCH_H


