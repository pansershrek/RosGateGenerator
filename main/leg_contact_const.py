import copy

init = [False, False, False, False]

II_ = copy.deepcopy(init)
PI_ = copy.deepcopy(init)
bI_ = copy.deepcopy(init)
IP_ = copy.deepcopy(init)
Ib_ = copy.deepcopy(init)
Pb_ = copy.deepcopy(init)
bP_ = copy.deepcopy(init)
BI_ = copy.deepcopy(init)
IB_ = copy.deepcopy(init)
PP_ = copy.deepcopy(init)
bb_ = copy.deepcopy(init)
Bb_ = copy.deepcopy(init)
BP_ = copy.deepcopy(init)
bB_ = copy.deepcopy(init)
PB_ = copy.deepcopy(init)
BB_ = copy.deepcopy(init)

# enum QuadrupedIDs { LF, RF, LH, RH };
LF = 0
RF = 1
LH = 2
RH = 3

PI_[LH] = True;
bI_[RH] = True;
IP_[LF] = True;
Ib_[RF] = True;

Pb_[LH] = True; Pb_[RF] = True;
bP_[RH] = True; bP_[LF] = True;
BI_[LH] = True; BI_[RH] = True;
IB_[LF] = True; IB_[RF] = True;
PP_[LH] = True; PP_[LF] = True;
bb_[RH] = True; bb_[RF] = True;

Bb_[LH] = True; Bb_[RH] = True;  Bb_[RF]= True;
BP_[LH] = True; BP_[RH] = True;  BP_[LF]= True;
bB_[RH] = True; bB_[LF] = True;  bB_[RF]= True;
PB_[LH] = True; PB_[LF] = True;  PB_[RF]= True;

BB_ = [True, True, True, True];

Stand = [
    BB_
]

Walk2 = [
  bB_, bb_, Bb_,
  Pb_,
  PB_, PP_, BP_,
  bP_,
]
Walk2E = [
  bB_, bb_, Bb_,
  Pb_,
  PB_, PP_, BP_
]
