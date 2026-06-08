# Rail-OCS3 DSL for Agg-OCS (B1): internal OCS ports connect ToR uplinks <-> trunk-to-core-ocs wires.
#
# This file is for humans. To compile into .schedule:
#   python3 /data/csg-htsim/sim/datacenter/rail_ocs3_schedule_dsl.py \
#     --in  /data/csg-htsim/sim/datacenter/topologies/agg_ocs_tor_trunk_pod256_tu32_trunk256.dsl \
#     --out /data/csg-htsim/sim/datacenter/topologies/agg_ocs_tor_trunk_pod256_tu32_trunk256.from_dsl.schedule
#
kind=agg  # rail-ocs3 Agg-OCS
tors=8
tor_up=32
trunk_ports=256

# plane p: connect each tor's uplink up=p to trunk lane (p*8+tor)
plane 0:
  tor(0).up(0) <-> trunk(0)
  tor(1).up(0) <-> trunk(1)
  tor(2).up(0) <-> trunk(2)
  tor(3).up(0) <-> trunk(3)
  tor(4).up(0) <-> trunk(4)
  tor(5).up(0) <-> trunk(5)
  tor(6).up(0) <-> trunk(6)
  tor(7).up(0) <-> trunk(7)

plane 1:
  tor(0).up(1) <-> trunk(8)
  tor(1).up(1) <-> trunk(9)
  tor(2).up(1) <-> trunk(10)
  tor(3).up(1) <-> trunk(11)
  tor(4).up(1) <-> trunk(12)
  tor(5).up(1) <-> trunk(13)
  tor(6).up(1) <-> trunk(14)
  tor(7).up(1) <-> trunk(15)

plane 2:
  tor(0).up(2) <-> trunk(16)
  tor(1).up(2) <-> trunk(17)
  tor(2).up(2) <-> trunk(18)
  tor(3).up(2) <-> trunk(19)
  tor(4).up(2) <-> trunk(20)
  tor(5).up(2) <-> trunk(21)
  tor(6).up(2) <-> trunk(22)
  tor(7).up(2) <-> trunk(23)

plane 3:
  tor(0).up(3) <-> trunk(24)
  tor(1).up(3) <-> trunk(25)
  tor(2).up(3) <-> trunk(26)
  tor(3).up(3) <-> trunk(27)
  tor(4).up(3) <-> trunk(28)
  tor(5).up(3) <-> trunk(29)
  tor(6).up(3) <-> trunk(30)
  tor(7).up(3) <-> trunk(31)

plane 4:
  tor(0).up(4) <-> trunk(32)
  tor(1).up(4) <-> trunk(33)
  tor(2).up(4) <-> trunk(34)
  tor(3).up(4) <-> trunk(35)
  tor(4).up(4) <-> trunk(36)
  tor(5).up(4) <-> trunk(37)
  tor(6).up(4) <-> trunk(38)
  tor(7).up(4) <-> trunk(39)

plane 5:
  tor(0).up(5) <-> trunk(40)
  tor(1).up(5) <-> trunk(41)
  tor(2).up(5) <-> trunk(42)
  tor(3).up(5) <-> trunk(43)
  tor(4).up(5) <-> trunk(44)
  tor(5).up(5) <-> trunk(45)
  tor(6).up(5) <-> trunk(46)
  tor(7).up(5) <-> trunk(47)

plane 6:
  tor(0).up(6) <-> trunk(48)
  tor(1).up(6) <-> trunk(49)
  tor(2).up(6) <-> trunk(50)
  tor(3).up(6) <-> trunk(51)
  tor(4).up(6) <-> trunk(52)
  tor(5).up(6) <-> trunk(53)
  tor(6).up(6) <-> trunk(54)
  tor(7).up(6) <-> trunk(55)

plane 7:
  tor(0).up(7) <-> trunk(56)
  tor(1).up(7) <-> trunk(57)
  tor(2).up(7) <-> trunk(58)
  tor(3).up(7) <-> trunk(59)
  tor(4).up(7) <-> trunk(60)
  tor(5).up(7) <-> trunk(61)
  tor(6).up(7) <-> trunk(62)
  tor(7).up(7) <-> trunk(63)

plane 8:
  tor(0).up(8) <-> trunk(64)
  tor(1).up(8) <-> trunk(65)
  tor(2).up(8) <-> trunk(66)
  tor(3).up(8) <-> trunk(67)
  tor(4).up(8) <-> trunk(68)
  tor(5).up(8) <-> trunk(69)
  tor(6).up(8) <-> trunk(70)
  tor(7).up(8) <-> trunk(71)

plane 9:
  tor(0).up(9) <-> trunk(72)
  tor(1).up(9) <-> trunk(73)
  tor(2).up(9) <-> trunk(74)
  tor(3).up(9) <-> trunk(75)
  tor(4).up(9) <-> trunk(76)
  tor(5).up(9) <-> trunk(77)
  tor(6).up(9) <-> trunk(78)
  tor(7).up(9) <-> trunk(79)

plane 10:
  tor(0).up(10) <-> trunk(80)
  tor(1).up(10) <-> trunk(81)
  tor(2).up(10) <-> trunk(82)
  tor(3).up(10) <-> trunk(83)
  tor(4).up(10) <-> trunk(84)
  tor(5).up(10) <-> trunk(85)
  tor(6).up(10) <-> trunk(86)
  tor(7).up(10) <-> trunk(87)

plane 11:
  tor(0).up(11) <-> trunk(88)
  tor(1).up(11) <-> trunk(89)
  tor(2).up(11) <-> trunk(90)
  tor(3).up(11) <-> trunk(91)
  tor(4).up(11) <-> trunk(92)
  tor(5).up(11) <-> trunk(93)
  tor(6).up(11) <-> trunk(94)
  tor(7).up(11) <-> trunk(95)

plane 12:
  tor(0).up(12) <-> trunk(96)
  tor(1).up(12) <-> trunk(97)
  tor(2).up(12) <-> trunk(98)
  tor(3).up(12) <-> trunk(99)
  tor(4).up(12) <-> trunk(100)
  tor(5).up(12) <-> trunk(101)
  tor(6).up(12) <-> trunk(102)
  tor(7).up(12) <-> trunk(103)

plane 13:
  tor(0).up(13) <-> trunk(104)
  tor(1).up(13) <-> trunk(105)
  tor(2).up(13) <-> trunk(106)
  tor(3).up(13) <-> trunk(107)
  tor(4).up(13) <-> trunk(108)
  tor(5).up(13) <-> trunk(109)
  tor(6).up(13) <-> trunk(110)
  tor(7).up(13) <-> trunk(111)

plane 14:
  tor(0).up(14) <-> trunk(112)
  tor(1).up(14) <-> trunk(113)
  tor(2).up(14) <-> trunk(114)
  tor(3).up(14) <-> trunk(115)
  tor(4).up(14) <-> trunk(116)
  tor(5).up(14) <-> trunk(117)
  tor(6).up(14) <-> trunk(118)
  tor(7).up(14) <-> trunk(119)

plane 15:
  tor(0).up(15) <-> trunk(120)
  tor(1).up(15) <-> trunk(121)
  tor(2).up(15) <-> trunk(122)
  tor(3).up(15) <-> trunk(123)
  tor(4).up(15) <-> trunk(124)
  tor(5).up(15) <-> trunk(125)
  tor(6).up(15) <-> trunk(126)
  tor(7).up(15) <-> trunk(127)

plane 16:
  tor(0).up(16) <-> trunk(128)
  tor(1).up(16) <-> trunk(129)
  tor(2).up(16) <-> trunk(130)
  tor(3).up(16) <-> trunk(131)
  tor(4).up(16) <-> trunk(132)
  tor(5).up(16) <-> trunk(133)
  tor(6).up(16) <-> trunk(134)
  tor(7).up(16) <-> trunk(135)

plane 17:
  tor(0).up(17) <-> trunk(136)
  tor(1).up(17) <-> trunk(137)
  tor(2).up(17) <-> trunk(138)
  tor(3).up(17) <-> trunk(139)
  tor(4).up(17) <-> trunk(140)
  tor(5).up(17) <-> trunk(141)
  tor(6).up(17) <-> trunk(142)
  tor(7).up(17) <-> trunk(143)

plane 18:
  tor(0).up(18) <-> trunk(144)
  tor(1).up(18) <-> trunk(145)
  tor(2).up(18) <-> trunk(146)
  tor(3).up(18) <-> trunk(147)
  tor(4).up(18) <-> trunk(148)
  tor(5).up(18) <-> trunk(149)
  tor(6).up(18) <-> trunk(150)
  tor(7).up(18) <-> trunk(151)

plane 19:
  tor(0).up(19) <-> trunk(152)
  tor(1).up(19) <-> trunk(153)
  tor(2).up(19) <-> trunk(154)
  tor(3).up(19) <-> trunk(155)
  tor(4).up(19) <-> trunk(156)
  tor(5).up(19) <-> trunk(157)
  tor(6).up(19) <-> trunk(158)
  tor(7).up(19) <-> trunk(159)

plane 20:
  tor(0).up(20) <-> trunk(160)
  tor(1).up(20) <-> trunk(161)
  tor(2).up(20) <-> trunk(162)
  tor(3).up(20) <-> trunk(163)
  tor(4).up(20) <-> trunk(164)
  tor(5).up(20) <-> trunk(165)
  tor(6).up(20) <-> trunk(166)
  tor(7).up(20) <-> trunk(167)

plane 21:
  tor(0).up(21) <-> trunk(168)
  tor(1).up(21) <-> trunk(169)
  tor(2).up(21) <-> trunk(170)
  tor(3).up(21) <-> trunk(171)
  tor(4).up(21) <-> trunk(172)
  tor(5).up(21) <-> trunk(173)
  tor(6).up(21) <-> trunk(174)
  tor(7).up(21) <-> trunk(175)

plane 22:
  tor(0).up(22) <-> trunk(176)
  tor(1).up(22) <-> trunk(177)
  tor(2).up(22) <-> trunk(178)
  tor(3).up(22) <-> trunk(179)
  tor(4).up(22) <-> trunk(180)
  tor(5).up(22) <-> trunk(181)
  tor(6).up(22) <-> trunk(182)
  tor(7).up(22) <-> trunk(183)

plane 23:
  tor(0).up(23) <-> trunk(184)
  tor(1).up(23) <-> trunk(185)
  tor(2).up(23) <-> trunk(186)
  tor(3).up(23) <-> trunk(187)
  tor(4).up(23) <-> trunk(188)
  tor(5).up(23) <-> trunk(189)
  tor(6).up(23) <-> trunk(190)
  tor(7).up(23) <-> trunk(191)

plane 24:
  tor(0).up(24) <-> trunk(192)
  tor(1).up(24) <-> trunk(193)
  tor(2).up(24) <-> trunk(194)
  tor(3).up(24) <-> trunk(195)
  tor(4).up(24) <-> trunk(196)
  tor(5).up(24) <-> trunk(197)
  tor(6).up(24) <-> trunk(198)
  tor(7).up(24) <-> trunk(199)

plane 25:
  tor(0).up(25) <-> trunk(200)
  tor(1).up(25) <-> trunk(201)
  tor(2).up(25) <-> trunk(202)
  tor(3).up(25) <-> trunk(203)
  tor(4).up(25) <-> trunk(204)
  tor(5).up(25) <-> trunk(205)
  tor(6).up(25) <-> trunk(206)
  tor(7).up(25) <-> trunk(207)

plane 26:
  tor(0).up(26) <-> trunk(208)
  tor(1).up(26) <-> trunk(209)
  tor(2).up(26) <-> trunk(210)
  tor(3).up(26) <-> trunk(211)
  tor(4).up(26) <-> trunk(212)
  tor(5).up(26) <-> trunk(213)
  tor(6).up(26) <-> trunk(214)
  tor(7).up(26) <-> trunk(215)

plane 27:
  tor(0).up(27) <-> trunk(216)
  tor(1).up(27) <-> trunk(217)
  tor(2).up(27) <-> trunk(218)
  tor(3).up(27) <-> trunk(219)
  tor(4).up(27) <-> trunk(220)
  tor(5).up(27) <-> trunk(221)
  tor(6).up(27) <-> trunk(222)
  tor(7).up(27) <-> trunk(223)

plane 28:
  tor(0).up(28) <-> trunk(224)
  tor(1).up(28) <-> trunk(225)
  tor(2).up(28) <-> trunk(226)
  tor(3).up(28) <-> trunk(227)
  tor(4).up(28) <-> trunk(228)
  tor(5).up(28) <-> trunk(229)
  tor(6).up(28) <-> trunk(230)
  tor(7).up(28) <-> trunk(231)

plane 29:
  tor(0).up(29) <-> trunk(232)
  tor(1).up(29) <-> trunk(233)
  tor(2).up(29) <-> trunk(234)
  tor(3).up(29) <-> trunk(235)
  tor(4).up(29) <-> trunk(236)
  tor(5).up(29) <-> trunk(237)
  tor(6).up(29) <-> trunk(238)
  tor(7).up(29) <-> trunk(239)

plane 30:
  tor(0).up(30) <-> trunk(240)
  tor(1).up(30) <-> trunk(241)
  tor(2).up(30) <-> trunk(242)
  tor(3).up(30) <-> trunk(243)
  tor(4).up(30) <-> trunk(244)
  tor(5).up(30) <-> trunk(245)
  tor(6).up(30) <-> trunk(246)
  tor(7).up(30) <-> trunk(247)

plane 31:
  tor(0).up(31) <-> trunk(248)
  tor(1).up(31) <-> trunk(249)
  tor(2).up(31) <-> trunk(250)
  tor(3).up(31) <-> trunk(251)
  tor(4).up(31) <-> trunk(252)
  tor(5).up(31) <-> trunk(253)
  tor(6).up(31) <-> trunk(254)
  tor(7).up(31) <-> trunk(255)
