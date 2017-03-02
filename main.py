import os

from utils import build_distort_p, load_distort_p

DISTORT_P = 'distort.p'

if not os.path.isfile(DISTORT_P):
    build_distort_p(DISTORT_P)

mtx, dist = load_distort_p(DISTORT_P)
