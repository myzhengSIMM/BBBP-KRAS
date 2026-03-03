import logd_scorer as LOGD
import bbb_scorer as BBB
import drd2_scorer as CCR5


def ccr5(s):
    m = CCR5.get_ccr5_score(s)
    print(m)
    return 0


def bbb(s):
    g = BBB.get_bbb_score(s)
    print(g)
    return 0


def logd(s):
    h = LOGD.get_logd_score(s)
    print(h)
    return 0


a = "CC1=NN=C(C(C)C)N1[C@H](C[C@H]2CC3)C[C@H]3N2CC[C@H](NC(C4CCC(F)(F)CC4)=O)C5=CSC=C5"
b = "O=C(C1CCC(F)(F)CC1)N[C@H](C2=CC=CC=C2)CCN3[C@H]4C[C@@H](N5C(C)=NN=C5C(C)C)C[C@@H]3CC4"
ccr5(a)
bbb(a)
logd(a)
ccr5(b)
bbb(b)
logd(b)
