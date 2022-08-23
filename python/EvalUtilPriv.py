#!/usr/bin/env python3
import numpy as np
from ot.lp import emd2_1d
from ot.sliced import get_random_projections
from ot.sliced import sliced_wasserstein_distance
from scipy.sparse import lil_matrix
from scipy.stats import wasserstein_distance
import math
import csv
import sys
import glob
import os

################################# Parameters ##################################
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "PPMTF", 10, "SmplA1"]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "PPMTF", 10, "SmplA2"]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "PPMTF", 10, "A"]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "SGD", 10]
#sys.argv = ["EvalUtilPriv.py", "PF", "TK", "SGLT", 10]

if len(sys.argv) < 4:
    print("Usage:",sys.argv[0],"[Dataset] [City] [SynAlg (PPMTF/PPITF/SGD/SGLT)] ([TraceNum (default:10)] [ParamA (default:A)] [ItrNum (default:100)] [PDTest (default:1)] [Reqk (default:10)])")
    sys.exit(0)

# Dataset (PF/FS)
DataSet = sys.argv[1]
# City
City = sys.argv[2]

# Synthesizing algorithm
SynAlg = sys.argv[3]

# Number of traces per user
TraceNum = 10
if len(sys.argv) >= 5:
    TraceNum = int(sys.argv[4])

# ParamA
ParamA = "A"
if len(sys.argv) >= 6:
    ParamA = sys.argv[5]

# Number of iterations in Gibbs sampling (SynAlg = "PPMTF")
ItrNum = 100
if len(sys.argv) >= 7:
    ItrNum = int(sys.argv[6])

# Perform the PD test (1: yes, 0: no)
PDTest = 1
if len(sys.argv) >= 8:
    PDTest = int(sys.argv[7])

# Required k in plausible deniability
Reqk = 10
if len(sys.argv) >= 9:
    Reqk = int(sys.argv[8])

# Data directory
DataDir = "../data/" + DataSet + "/"

# Training user index file (input)
TUserIndexFile = DataDir + "tuserindex_XX.csv"
# Testing user index file (output)
EUserIndexFile = DataDir + "euserindex_XX.csv"
# POI file (input)
POIFile = DataDir + "POI_XX.csv"
# POI index file (input)
POIIndexFile = DataDir + "POIindex_XX.csv"
# Training trace file (input)
TrainTraceFile = DataDir + "traintraces_XX.csv"
# Testing trace file (input)
TestTraceFile = DataDir + "testtraces_XX.csv"

# Type of time slots (1: 9-19h, 20min, 2: 2 hours)
if DataSet == "PF":
    TimeType = 1
elif DataSet[0:2] == "FS":
    TimeType = 2
else:
    print("Wrong Dataset")
    sys.exit(-1)

# Maximum time interval between two temporally-continuous locations (sec) (-1: none)
if DataSet == "PF":
    MaxTimInt = -1
elif DataSet[0:2] == "FS":
    MaxTimInt = 7200

# Number of columns in model parameters (A, B, C)
K = 16

# Minimum probability
DeltaProb = 0.00000001
# Top-L POIs (Time-specific)
L1 = 50
# Number of bins in the visit-fraction distribution
if DataSet == "PF":
    B = 30
elif DataSet[0:2] == "FS":
    B = 24
else:
    print("Wrong Dataset")
    sys.exit(-1)

# Minimum/maximum of y/x
if DataSet == "PF" and City == "TK":
    MIN_Y = 35.65
    MAX_Y = 35.75
    MIN_X = 139.68
    MAX_X = 139.8
elif DataSet[0:2] == "FS" and City == "IS":
    MIN_Y = 40.8
    MAX_Y = 41.2
    MIN_X = 28.5
    MAX_X = 29.5
elif DataSet[0:2] == "FS" and City == "JK":
    MIN_Y = -6.4
    MAX_Y = -6.1
    MIN_X = 106.7
    MAX_X = 107.0
elif DataSet[0:2] == "FS" and City == "KL":
    MIN_Y = 3.0
    MAX_Y = 3.3
    MIN_X = 101.6
    MAX_X = 101.8
elif DataSet[0:2] == "FS" and City == "NY":
    MIN_Y = 40.5
    MAX_Y = 41.0
    MIN_X = -74.28
    MAX_X = -73.68
elif DataSet[0:2] == "FS" and City == "TK":
    MIN_Y = 35.5
    MAX_Y = 35.9
    MIN_X = 139.5
    MAX_X = 140.0
elif DataSet[0:2] == "FS" and City == "SP":
    MIN_Y = -24.0
    MAX_Y = -23.4
    MIN_X = -46.8
    MAX_X = -46.3
else:
    print("Wrong Dataset")
    sys.exit(-1)
# Number of regions on the x-axis
NumRegX = 20
# Number of regions on the y-axis
NumRegY = 20

if SynAlg == "PPMTF":
    # Prefix of the synthesized trace directory
    SynTraceDirPre = DataDir + "PPMTF_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "*/"
    # Synthesized trace file (with asterisk)
    if DataSet == "PF" and ParamA == "A":
        SynTraceFileAst = SynTraceDir + "syntraces_Itr" + str(ItrNum) + ".csv"
    else:
        SynTraceFileAst = SynTraceDir + "syntraces_Itr" + str(ItrNum) + "_" + ParamA + ".csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_PPMTF_" + City + "_" + ParamA + ".csv"
elif SynAlg == "PPITF":
    # Prefix of the synthesized trace directory
    SynTraceDirPre = DataDir + "PPITF_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "*/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "syntraces_Itr" + str(ItrNum) + ".csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_PPITF_" + City + ".csv"
elif SynAlg == "SGD":
    # Prefix of the synthesized trace directory  
    SynTraceDirPre = DataDir + "SGD_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "syntraces_cn*.csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_SGD_" + City + ".csv"
elif SynAlg == "SGLT":
    # Prefix of the synthesized trace directory  
    SynTraceDirPre = DataDir + "SGLT_" + City
    # Synthesized trace directory
    SynTraceDir = SynTraceDirPre + "/"
    # Synthesized trace file (with asterisk)
    SynTraceFileAst = SynTraceDir + "*_syntraces.csv"
    # Result file (output)
    ResFile = DataDir + "utilpriv_SGLT_" + City + ".csv"
else:
    print("Wrong SynAlg")
    sys.exit(-1)

# Prefix of the model parameter file (input)
ModelParameterDir = DataDir + "PPMTF_" + City + "_alp200_mnt100_mnv100/"
ModelParameterFile = ModelParameterDir + "modelparameter"

# The following function was used for POT-0.8.0.dev0 (it is not useful for POT-0.8.2)
def sliced_wasserstein_distance_p1(X_s, X_t, a=None, b=None, n_projections=50, seed=None, log=False):
    X_s = np.asanyarray(X_s)
    X_t = np.asanyarray(X_t)

    n = X_s.shape[0]
    m = X_t.shape[0]

    if X_s.shape[1] != X_t.shape[1]:
        raise ValueError(
            "X_s and X_t must have the same number of dimensions {} and {} respectively given".format(X_s.shape[1],
                                                                                                      X_t.shape[1]))

    if a is None:
        a = np.full(n, 1 / n)
    if b is None:
        b = np.full(m, 1 / m)

    d = X_s.shape[1]

    projections = get_random_projections(n_projections, d, seed)

    X_s_projections = np.dot(projections, X_s.T)
    X_t_projections = np.dot(projections, X_t.T)

    if log:
        projected_emd = np.empty(n_projections)
    else:
        projected_emd = None

    res = 0.

    for i, (X_s_proj, X_t_proj) in enumerate(zip(X_s_projections, X_t_projections)):
#        emd = emd2_1d(X_s_proj, X_t_proj, a, b, log=False, dense=False)
        emd = emd2_1d(X_s_proj, X_t_proj, a, b, metric='minkowski', p=1.0, log=False, dense=False)
        if projected_emd is not None:
            projected_emd[i] = emd
        res += emd

#    res = (res / n_projections) ** 0.5
    res = res / n_projections
    if log:
        return res, {"projections": projections, "projected_emds": projected_emd}
    return res


######################### Read a training trace file ##########################
# [output1]: ttrans_count ({(user_index, poi_index_from, poi_index_to): counts})
# [output2]: ttrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [output3]: tcount_sum (N x M matrix)
def ReadTrainTraceFile():
    # Initialization
    ttrans_count = {}
    ttrans_prob = {}
    tcount_sum = np.zeros((N, M))
    user_index_prev = -1
    poi_index_prev = 0

    # Read a training trace file
    f = open(TrainTraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0])
        poi_index = int(lst[1])
        # Update a transition matrix if the event and the previous event are from the same user
        if user_index == user_index_prev:
            ttrans_count[(user_index, poi_index_prev, poi_index)] = ttrans_count.get((user_index, poi_index_prev, poi_index), 0) + 1
        user_index_prev = user_index
        poi_index_prev = poi_index
    f.close()

    # Make a count sum matrix --> tcount_sum
    for (user_index, poi_index_prev, poi_index), counts in sorted(ttrans_count.items()):
        tcount_sum[user_index, poi_index_prev] += counts

    # Make a transition probability tensor --> ttrans_prob
    for (user_index, poi_index_prev, poi_index), counts in sorted(ttrans_count.items()):
        ttrans_prob[(user_index, poi_index_prev, poi_index)] = counts / tcount_sum[user_index, poi_index_prev]

    return ttrans_count, ttrans_prob, tcount_sum

######################### Read a testing trace file ##########################
# [output1]: etrans_count ({(user_index, poi_index_from, poi_index_to): counts})
# [output2]: etrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [output3]: ecount_sum (N x M matrix)
def ReadTestTraceFile():
    # Initialization
    etrans_count = {}
    etrans_prob = {}
    ecount_sum = np.zeros((N, M))
    user_index_prev = -1
    poi_index_prev = 0

    # Read a testing trace file
    f = open(TestTraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0]) - N
        poi_index = int(lst[1])
        # Update a transition matrix if the event and the previous event are from the same user
        if user_index == user_index_prev:
            etrans_count[(user_index, poi_index_prev, poi_index)] = etrans_count.get((user_index, poi_index_prev, poi_index), 0) + 1
        user_index_prev = user_index
        poi_index_prev = poi_index
    f.close()

    # Make a count sum matrix --> ecount_sum
    for (user_index, poi_index_prev, poi_index), counts in sorted(etrans_count.items()):
        ecount_sum[user_index, poi_index_prev] += counts

    # Make a transition probability tensor --> etrans_prob
    for (user_index, poi_index_prev, poi_index), counts in sorted(etrans_count.items()):
        etrans_prob[(user_index, poi_index_prev, poi_index)] = counts / ecount_sum[user_index, poi_index_prev]

    return etrans_count, etrans_prob, ecount_sum

######################## MAP re-identification attack #########################
# [input1]: ttrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [input2]: tcount_sum (N x M matrix)
# [input3]: syn_trace_file
def MAPReidentify(ttrans_prob, tcount_sum, syn_trace_file):
    # Initialization
    log_post = np.zeros(N)
    reid_res = np.zeros(N*TraceNum)
    user_index_prev = 0
    poi_index_prev = 0

    # Read a synthesized trace file
    f = open(syn_trace_file, "r")
    reader = csv.reader(f)
    next(reader)
    user_no = 0
    time_slot = 0
    for lst in reader:
        user_index = int(lst[0])

        trace_index = int(lst[1])
        user_index = user_index * TraceNum + trace_index

        poi_index = int(lst[4])

        if user_index != user_index_prev:
            time_slot = 0

        # Update the log-posterior if the event and the previous event are from the same user
        if user_index == user_index_prev and time_slot >= 1:
            # For each user
            for n in range(N):
                if tcount_sum[n,poi_index_prev] > 0:
                    if (n, poi_index_prev, poi_index) in ttrans_prob:
                        log_post[n] += math.log(ttrans_prob[n,poi_index_prev,poi_index])
                    else:
                        log_post[n] += math.log(DeltaProb)
                else:
                    log_post[n] += math.log(DeltaProb)
        # Update the re-identification result if a new user appears --> reid_res
        elif user_index != user_index_prev:
            reid_res[user_no] = np.argmax(log_post)
            log_post = np.zeros(N)
            user_no += 1

        user_index_prev = user_index
        poi_index_prev = poi_index
        time_slot += 1
    f.close()

    # Update the re-identification result for the last user
    reid_res[user_no] = np.argmax(log_post)

    return log_post, reid_res

############# Likelihood-ratio-based membership inference attack ##############
# [input1]: ttrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [input2]: etrans_prob ({(user_index, poi_index_from, poi_index_to): probability})
# [input3]: syn_trace_file
# [output1]: llr_per_trace ((TraceNum x (N+N2) matrix)
# [output2]: trace_thr (1000-dim vector)
# [output3]: trace_true_pos (1000-dim vector)
# [output4]: trace_true_neg (1000-dim vector)
# [output5]: trace_max_acc
# [output6]: trace_max_adv
def LRMIA(ttrans_prob, etrans_prob, syn_trace_file):
    # Initialization
    llr_per_trace = np.full((TraceNum, N+N2), -sys.float_info.max)

    # Membership inference for each training/testing user n
    for n in range(N+N2):
        # Population transition probability matrix --> pop_trans_prob
        pop_trans_prob = np.zeros((M, M))
        # Population transition matrix except for training user n
        for (user_index, poi_index_prev, poi_index), prob in sorted(ttrans_prob.items()):
            if n < N and user_index == n:
                continue
            pop_trans_prob[poi_index_prev, poi_index] += prob
        # Population transition matrix except for testing user n-N
        for (user_index, poi_index_prev, poi_index), prob in sorted(etrans_prob.items()):
            if n >= N and user_index == n-N:
                continue
            pop_trans_prob[poi_index_prev, poi_index] += prob
        pop_trans_prob /= (N+N2-1)

        # Read a synthesized trace file
        f = open(syn_trace_file, "r")
        reader = csv.reader(f)
        next(reader)
        
        # Initialization
        user_index_prev = 0
        poi_index_prev = 0
        time_slot = 0
        llr_trace = np.zeros(TraceNum)

        for lst in reader:
            user_index = int(lst[0])
            trace_index = int(lst[1])
#            user_index = user_index * TraceNum + trace_index
    
            poi_index = int(lst[4])
    
            if user_index != user_index_prev:
                time_slot = 0
    
            # Update the log-likelihood ratio if the event and the previous event are from the same user
            if user_index == user_index_prev and time_slot >= 1:
                # Update the log-likelihood ratio for the user --> llr_trace[trace_index]
                # Membership inference for training user n
                if n < N:
                    # Add the log-likelihood using the transition matrix of training user n
                    if (n, poi_index_prev, poi_index) in ttrans_prob:
                        llr_trace[trace_index] += math.log(ttrans_prob[n,poi_index_prev,poi_index])
                    else:
                        llr_trace[trace_index] += math.log(DeltaProb)
                # Membership inference for testing user n-N
                else:
                    # Add the log-likelihood using the transition matrix of testing user n-N
                    if (n-N, poi_index_prev, poi_index) in etrans_prob:
                        llr_trace[trace_index] += math.log(etrans_prob[n-N,poi_index_prev,poi_index])
                    else:
                        llr_trace[trace_index] += math.log(DeltaProb)
                # Subtract the log-likelihood using the population matrix
                if pop_trans_prob[poi_index_prev, poi_index] > 0:
                    llr_trace[trace_index] -= math.log(pop_trans_prob[poi_index_prev, poi_index])
                else:
                    llr_trace[trace_index] -= math.log(DeltaProb)

            # Update llr_per_trace if a new user appears
            elif user_index != user_index_prev:
                for tr in range(TraceNum):
                    if llr_per_trace[tr,n] < llr_trace[tr]:
                        llr_per_trace[tr,n] = llr_trace[tr]
                llr_trace = np.zeros(TraceNum)
    
            user_index_prev = user_index
            poi_index_prev = poi_index
            time_slot += 1
        f.close()

        # Update the log-likelihood ratio for the last user
        for tr in range(TraceNum):
            if llr_per_trace[tr,n] < llr_trace[tr]:
                llr_per_trace[tr,n] = llr_trace[tr]

    # Calculate #true positive/negative using llr_per_trace --> trace_true_pos, trace_true_neg
    MIA_thr = np.zeros(1000)
    max_llr_per_trace = -sys.float_info.max
    min_llr_per_trace = -sys.float_info.max
    for tr in range(TraceNum):
        if max_llr_per_trace < max(llr_per_trace[tr]):
            max_llr_per_trace = max(llr_per_trace[tr])
        if min_llr_per_trace < min(llr_per_trace[tr]):
            min_llr_per_trace = min(llr_per_trace[tr])
    MIA_true_pos = np.zeros(1000)
    MIA_true_neg = np.zeros(1000)
    # For each threshold
    for i in range(1000):
        # Threshold --> thr
        MIA_thr[i] = min_llr_per_trace + (max_llr_per_trace - min_llr_per_trace) * i / 1000
        # True positive --> true_pos
        for tr in range(TraceNum):
            for n in range(N):
                if llr_per_trace[tr,n] > MIA_thr[i]:
                    MIA_true_pos[i] += 1
            # True negative --> true_neg
            for n in range(N2):
                if llr_per_trace[tr,N+n] <= MIA_thr[i]:
                    MIA_true_neg[i] += 1
    # Calculate the maximum accuracy using llr_per_trace --> MIA_max_acc
    MIA_max_acc = 0
    for i in range(1000):
        if MIA_max_acc < MIA_true_pos[i] + MIA_true_neg[i]:
            MIA_max_acc = MIA_true_pos[i] + MIA_true_neg[i]
    MIA_max_acc /= (TraceNum*(N+N2))

    # Calculate the maximum membership advantage using llr_per_trace --> MIA_max_adv
    MIA_max_adv = -sys.float_info.max
    for i in range(1000):
        if MIA_max_adv < MIA_true_pos[i]/(TraceNum*N) - 1 + MIA_true_neg[i]/(TraceNum*N2):
            MIA_max_adv = MIA_true_pos[i]/(TraceNum*N) - 1 + MIA_true_neg[i]/(TraceNum*N2)

    return llr_per_trace, MIA_thr, MIA_true_pos, MIA_true_neg, MIA_max_acc, MIA_max_adv

############################### Read POI files ################################
# [output1]: poi_dic ({poi_index: [y, x, y_id, x_id, category]})
def ReadPOI():
    # Initialization
    poi_dic = {}
    poi_file_dic = {}

    # Calculate the boundaries of the regions (NumRegX x NumRegY) --> xb, yb
    xb = np.zeros(NumRegX)
    yb = np.zeros(NumRegY)
    for i in range(NumRegX):
        xb[i] = MIN_X + (MAX_X - MIN_X) * i / NumRegX
    for i in range(NumRegY):
        yb[i] = MIN_Y + (MAX_Y - MIN_Y) * i / NumRegY
 
    # Read a POI file --> poi_file_dic ({poi_id: [y, x, y_id, x_id]})
    f = open(POIFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        y = float(lst[1])
        x = float(lst[2])

        x_id = NumRegX-1
        for i in range(NumRegX-1):
            if xb[i] <= x < xb[i+1]:
                x_id = i
                break
        y_id = NumRegY-1
        for i in range(NumRegY-1):
            if yb[i] <= y < yb[i+1]:
                y_id = i
                break

        poi_file_dic[lst[0]] = [y, x, y_id, x_id]
    f.close()

    # Read a POI index file --> poi_dic ({poi_index: [y, x, y_id, x_id, category]})
    f = open(POIIndexFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        y = poi_file_dic[lst[0]][0]
        x = poi_file_dic[lst[0]][1]
        y_id = poi_file_dic[lst[0]][2]
        x_id = poi_file_dic[lst[0]][3]
        poi_dic[int(lst[1])] = [y, x, y_id, x_id, lst[2]]
    f.close()

    return poi_dic

########################### Read model parameters #############################
# [output1]: ParamA (N x K matrix)
# [output2]: ParamB (M x K matrix)
# [output3]: ParamC (M x K matrix)
# [output4]: ParamD (T x K matrix)
def ReadModelParameters():
    # Read model parameter A
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_A.csv"
    f = open(infile, "r")
    ParamA = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter B
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_B.csv"
    f = open(infile, "r")
    ParamB = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter C
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_C.csv"
    f = open(infile, "r")
    ParamC = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter D
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_D.csv"
    f = open(infile, "r")
    ParamD = np.loadtxt(infile, delimiter=",")
    f.close()

    return ParamA, ParamB, ParamC, ParamD

############################## Read real traces ###############################
# [input1]: st_user_index -- Start user index
# [input2]: user_num -- Number of users
# [input3]: M -- Number of POIs
# [input4]: T -- Number of time slots
# [input5]: A_bin (N x K matrix)
# [input6]: trace_file -- Trace file
# [output1]: treal_dist (T x M matrix)
# [output2]: treal_count (T-dim vector)
# [output3]: real_trans (M x M matrix)
# [output4]: vf_dist (M x B matrix)
# [output5]: kreal_dist (K x (1 x M matrix))
# [output6]: ktreal_dist (K x (T x M matrix))
def ReadRealTraces(st_user_index, user_num, M, T, A_bin, trace_file):
    # Initialization
    ureal_visit = lil_matrix((user_num, M))
#    visitor_real_rate = np.zeros(M)
    vf = np.zeros(M)
    vf_dist = np.zeros((M, B))
    ureal_count = np.zeros(user_num)
    treal_dist = np.zeros((T, M))
    treal_count = np.zeros(T)
    real_trans = np.zeros((M, M))
    user_index_prev = -1
    poi_index_prev = 0
    unixtime_prev = 0
    time_ins_prev = 0
    kreal_dist = [0] * K
    for k in range(K):
        kreal_dist[k] = np.zeros((1, M))
    ktreal_dist = [0] * K
    for k in range(K):
        ktreal_dist[k] = np.zeros((T, M))

    # Read a real trace file
    f = open(trace_file, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0]) - st_user_index
        poi_index = int(lst[1])
        unixtime = float(lst[3])
        ho = int(lst[5])
        if TimeType == 1:
            if int(lst[6]) >= 40:
                mi = 2
            elif int(lst[6]) >= 20:
                mi = 1
            else:
                mi = 0
            time_slot = 3 * (ho - 9) + mi
            time_ins = time_slot
        elif TimeType == 2:
            time_slot = int(ho/2)
            time_ins = ho
        else:
            print("Wrong TimeType.\n")
            sys.exit(-1)

        # New user
        if user_index != user_index_prev and user_index_prev != -1:
            if ureal_count[user_index_prev] >= 5:
                # Normalize vf
                vf /= ureal_count[user_index_prev]
                # Update vf_dist
                for i in range(M):
                    # Continue if the visit-count is zero or one
                    if vf[i] == 0 or vf[i] == 1:
                        continue
                    vf_bin = math.ceil(vf[i] * B) - 1
                    vf_dist[i,vf_bin] += 1
            # Initialization
            vf = np.zeros(M)

        # Update the user-specific visit matrix --> ureal_visit
        ureal_visit[user_index, poi_index] = 1
        # Update the visit-fraction
        vf[poi_index] += 1

        # Update visit counts for the user --> ureal_count
        ureal_count[user_index] += 1
        # Update the time-specific real distribution --> treal_dist
        treal_dist[time_slot, poi_index] += 1

        # Update the time-specific real distribution for each basis --> kreal_dist, ktreal_dist
        for k in range(K):
            if A_bin[user_index, k] == True:
                kreal_dist[k][0, poi_index] += 1
                ktreal_dist[k][time_slot, poi_index] += 1

        # Update visit counts for the time instance --> treal_count
        treal_count[time_slot] += 1
        # Update a transition matrix if the event and the previous event are from the same user
        if user_index == user_index_prev and (MaxTimInt == -1 or unixtime - unixtime_prev <= MaxTimInt) and time_ins - time_ins_prev == 1:
            real_trans[poi_index_prev, poi_index] += 1
        user_index_prev = user_index
        poi_index_prev = poi_index
        unixtime_prev = unixtime
        time_ins_prev = time_ins
    f.close()

    # Last user
    if ureal_count[user_index_prev] >= 5:
        # Normalize the visit-fraction
        vf /= ureal_count[user_index_prev]
        # Update vf_dist
        for i in range(M):
            # Continue if the visit-count is zero or one
            if vf[i] == 0 or vf[i] == 1:
                continue
            vf_bin = math.ceil(vf[i] * B) - 1
            vf_dist[i,vf_bin] += 1

    # Normalize vf_dist
    for i in range(M):
        if np.sum(vf_dist[i]) > 0:
            vf_dist[i] /= np.sum(vf_dist[i])

    # Normalize treal_dist
    for t in range(T):
        if np.sum(treal_dist[t]) > 0:
            treal_dist[t] /= np.sum(treal_dist[t])
        else:
            treal_dist[t] = np.full(M, 1.0/float(M))

    # Normalize kreal_dist
    for k in range(K):
        if np.sum(kreal_dist[k][0]) > 0:
            kreal_dist[k][0] /= np.sum(kreal_dist[k][0])
        else:
            kreal_dist[k][0] = np.full(M, 1.0/float(M))

    # Normalize ktreal_dist
    for k in range(K):
        for t in range(T):
            if np.sum(ktreal_dist[k][t]) > 0:
                ktreal_dist[k][t] /= np.sum(ktreal_dist[k][t])
            else:
                ktreal_dist[k][t] = np.full(M, 1.0/float(M))

    # Normalize real_trans
    for i in range(M):
        if np.sum(real_trans[i]) > 0:
            real_trans[i] /= np.sum(real_trans[i])
        else:
            real_trans[i,i] = 1.0

    return treal_dist, treal_count, real_trans, vf_dist, kreal_dist, ktreal_dist

########################### Read synthesized traces ###########################
# [input1]: N -- Number of users
# [input2]: M -- Number of POIs
# [input3]: T -- Number of time slots
# [input4]: A_bin (N x K matrix)
# [input5]: syn_trace_file -- Synthesized trace file
# [input6]: pdtest_file -- PD test result file
# [input7]: req_k -- Required k
# [input8]: trace_no -- Trace no.
# [output1]: tsyn_dist (T x M matrix)
# [output2]: syn_trans (M x M matrix)
# [output3]: pass_test (N-dim vector)
# [output4]: vf_dist (M x B matrix)
# [output5]: ksyn_dist (K x (1 x M matrix))
# [output6]: ktsyn_dist (K x (T x M matrix))
def ReadSynTraces(N, M, T, A_bin, syn_trace_file, pdtest_file, req_k, trace_no):
    # Initialization
    usyn_visit = lil_matrix((N, M))
    vf = np.zeros(M)
    vf_dist = np.zeros((M, B))
    usyn_count = np.zeros(N)
    tsyn_dist = np.zeros((T, M))
    syn_trans = np.zeros((M, M))
    pass_test = np.ones(N)
    user_index_prev = -1
    poi_index_prev = 0
    ksyn_dist = [0] * K
    for k in range(K):
        ksyn_dist[k] = np.zeros((1, M))
    ktsyn_dist = [0] * K
    for k in range(K):
        ktsyn_dist[k] = np.zeros((T, M))

    # Read a PD test result file --> pass_test
    if pdtest_file != "none":
        infile = pdtest_file + "_Itr" + str(ItrNum) + ".csv"
        i = 0
        f = open(infile, "r")
        reader = csv.reader(f)
        for lst in reader:
            if lst[0] == "-":
                break
            k = float(lst[0])
            if k < req_k:
                pass_test[i] = 0
            i += 1
        print("Fraction of passing the PD test:", float(np.sum(pass_test)) / float(N), "(", np.sum(pass_test), "/", N, ")")

    # Read a real trace file
    f = open(syn_trace_file, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        user_index = int(lst[0])
        trace_no_cur = int(lst[1])
        time_slot = int(lst[2])
        poi_index = int(lst[4])

        if trace_no_cur != trace_no and trace_no != -1:
            continue

        if pass_test[user_index] == 1:
            # New user
            if user_index != user_index_prev and user_index_prev != -1:
                if usyn_count[user_index_prev] >= 5:
                    # Normalize vf
                    vf /= usyn_count[user_index_prev]
                    # Update vf_dist
                    for i in range(M):
                        # Continue if the visit-count is zero or one
                        if vf[i] == 0 or vf[i] == 1:
                            continue
                        vf_bin = math.ceil(vf[i] * B) - 1
                        vf_dist[i,vf_bin] += 1
                # Initialization
                vf = np.zeros(M)

            # Update the user-specific visit matrix --> usyn_visit
            usyn_visit[user_index, poi_index] = 1
            # Update the visit-fraction
            vf[poi_index] += 1
            # Update visit counts for the user --> usyn_count
            usyn_count[user_index] += 1
            # Update the time-specific synthesized distribution --> tsyn_dist
            tsyn_dist[time_slot, poi_index] += 1

            # Update the real distribution & time-specific real distribution for each basis --> ksyn_dist, ktsyn_dist
            for k in range(K):
                if A_bin[user_index, k] == True:
                    ksyn_dist[k][0, poi_index] += 1
                    ktsyn_dist[k][time_slot, poi_index] += 1

            # Update a transition matrix if the event and the previous event are from the same user
            if user_index == user_index_prev:
                syn_trans[poi_index_prev, poi_index] += 1
            user_index_prev = user_index
            poi_index_prev = poi_index
    f.close()

    # Last user
    if usyn_count[user_index_prev] >= 5:
        # Normalize the visit-fraction
        vf /= usyn_count[user_index_prev]
        # Update vf_dist
        for i in range(M):
            # Continue if the visit-count is zero or one
            if vf[i] == 0 or vf[i] == 1:
                continue
            vf_bin = math.ceil(vf[i] * B) - 1
            vf_dist[i,vf_bin] += 1

    # Normalize vf_dist
    for i in range(M):
        if np.sum(vf_dist[i]) > 0:
            vf_dist[i] /= np.sum(vf_dist[i])
        else:
            vf_dist[i,0] = 1.0

    # Normalize tsyn_dist
    for t in range(T):
        if np.sum(tsyn_dist[t]) > 0:
            tsyn_dist[t] /= np.sum(tsyn_dist[t])
        else:
            tsyn_dist[t] = np.full(M, 1.0/float(M))

    # Normalize ksyn_dist
    for k in range(K):
        if np.sum(ksyn_dist[k][0]) > 0:
            ksyn_dist[k][0] /= np.sum(ksyn_dist[k][0])
        else:
            ksyn_dist[k][0] = np.full(M, 1.0/float(M))

    # Normalize ktsyn_dist
    for k in range(K):
        for t in range(T):
            if np.sum(ktsyn_dist[k][t]) > 0:
                ktsyn_dist[k][t] /= np.sum(ktsyn_dist[k][t])
            else:
                ktsyn_dist[k][t] = np.full(M, 1.0/float(M))

    # Normalize syn_trans
    for i in range(M):
        if np.sum(syn_trans[i]) > 0:
            syn_trans[i] /= np.sum(syn_trans[i])
        else:
            syn_trans[i,i] = 1.0

    return tsyn_dist, syn_trans, pass_test, vf_dist, ksyn_dist, ktsyn_dist

############################# Calculate the SWD ###############################
# [input1]: dist1 (Z x M matrix)
# [input2]: dist2 (Z x M matrix)
# [input3]: Z -- Number of distributions
# [input4]: M -- Number of POIs
# [output1]: avg_swd -- Average SWD
def CalcSWD(dist1, dist2, Z, M):
    # Initializaion
    avg_swd = 0

    # data in the 2D map --> data_2D
    data_2D = np.zeros((M, 2))
    for i in range(M):
        y_id = poi_dic[i][2]
        x_id = poi_dic[i][3]
        data_2D[i] = (x_id, y_id)

    # For each distribution
    for z in range(Z):
        # i-th row (i-th distribution) in dist1 --> p1
        p1 = dist1[z,:]
        # i-th row (i-th distribution) in dist2 --> p2
        p2 = dist2[z,:]
        # Calculate the SWD
#        avg_swd += sliced_wasserstein_distance_p1(data_2D, data_2D, a=p1, b=p2, seed=0)
        avg_swd += sliced_wasserstein_distance(data_2D, data_2D, a=p1, b=p2, p=1, seed=0)

    avg_swd /= Z
    
    return avg_swd
            
#################### Calculate the average l1 & l2 losses #####################
# [input1]: dist1 (Z x M matrix)
# [input2]: dist2 (Z x M matrix)
# [input3]: pass_test (Z-dim vector)
# [input4]: Z -- Number of distributions
# [input5]: M -- Number of POIs
# [input6]: L -- Number of top POIs in dist1
# [output1]: l1_loss
# [output2]: l2_loss
def CalcL1L2(dist1, dist2, pass_test, Z, M, L):
    l1_loss = 0.0
    l2_loss = 0.0
    z_num = 0

    # l1 & l2 losses for all POIs
    if L == M:
        for z in range(Z):
            if pass_test[z] == 1:
                # Update the l1-loss & l2-loss
                for i in range(M):
                    l1_loss += np.abs(dist1[z,i] - dist2[z,i])
                    l2_loss += (dist1[z,i] - dist2[z,i])**2
                z_num += 1
    # l1 & l2 losses for the top L POIs
    else:
        for z in range(Z):
            if pass_test[z] == 1:
                # Sort indexes in descending order of dist1
                sortindex = np.argsort(dist1[z])[::-1]

                # Update the l1-loss & l2-loss
                for i in range(L):
                    j = sortindex[i]
                    l1_loss += np.abs(dist1[z,j] - dist2[z,j])
                    l2_loss += (dist1[z,j] - dist2[z,j])**2
                z_num += 1
        
    # Normalize l1_loss and l2_loss
    l1_loss /= z_num
    l2_loss /= z_num

    return l1_loss, l2_loss

####### Calculate the l1-loss using visit-fraction distributions ########
# [input1]: vf_dist1 (M x B matrix)
# [input2]: vf_dist2 (M x B matrix)
# [input3]: vf_exist (M-dim vector)
# [input4]: M -- Number of POIs
# [input5]: B -- Number of bins
# [output1]: l1_loss
def CalcL1VfDist(vf_dist1, vf_dist2, vf_exist, M, B):
    # Initialization
    l1_loss = 0
    x = 0

    # For each POI
    for i in range(M):
        # i-th row (visit-fraction for the i-th POI) in vf_dist1 --> p1
        p1 = vf_dist1[i,:]
        # i-th row (visit-fraction for the i-th POI) in vf_dist2 --> p2
        p2 = vf_dist2[i,:]
        # Continue if either testing vf or training vf doesn't exist
        if vf_exist[i] == 0:
            continue
        # l1-loss between p1 & p2 --> l1_loss
        for j in range(B):
            l1_loss += np.abs(p1[j] - p2[j])
        x += 1

    # Calculate the average l1-loss
    l1_loss /= x

    return l1_loss

################# Calculate the SWD using transition matrices #################
# [input1]: trans1 (M x M matrix)
# [input2]: trans2 (M x M matrix)
# [input3]: poi_dic ({poi_index: [y, x, y_id, x_id]})
# [input4]: M -- Number of POIs
# [output1]: avg_swd -- Average SWD
def CalcSWDTransMat(trans1, trans2, poi_dic, M):
    # Initializaion
    avg_swd = 0

    # data in the 2D map --> data_2D
    data_2D = np.zeros((M, 2))
    for i in range(M):
        y_id = poi_dic[i][2]
        x_id = poi_dic[i][3]
        data_2D[i] = (x_id, y_id)

    # For each POI
    for i in range(M):
        # i-th row (conditional probability from the i-th POI) in trans1 --> p1
        p1 = trans1[i,:]
        # i-th row (conditional probability from the i-th POI) in trans2 --> p2
        p2 = trans2[i,:]
        # Calculate the SWD
#        avg_swd += sliced_wasserstein_distance_p1(data_2D, data_2D, a=p1, b=p2, seed=0)
        avg_swd += sliced_wasserstein_distance(data_2D, data_2D, a=p1, b=p2, p=1, seed=0)
    avg_swd /= M

    return avg_swd
    
######### Calculate the EMD on the y/x-axis using transition matrices #########
# [input1]: trans1 (M x M matrix)
# [input2]: trans2 (M x M matrix)
# [input3]: poi_dic ({poi_index: [y, x, y_id, x_id]})
# [input4]: M -- Number of POIs
# [output1]: avg_emd_y -- Average EMD (y-axis)
# [output2]: avg_emd_x -- Average EMD (x-axis)
def CalcEMDTransMat_XY(trans1, trans2, poi_dic, M):
    # Initializaion
    avg_emd_y = 0
    avg_emd_x = 0
    y_axis_ids = np.arange(NumRegY)
    x_axis_ids = np.arange(NumRegX)

    # For each POI
    for i in range(M):
        # Initialization
        p1_y = np.zeros(NumRegY)
        p1_x = np.zeros(NumRegX)
        p2_y = np.zeros(NumRegY)
        p2_x = np.zeros(NumRegX)

        # i-th row (conditional probability from the i-th POI) in trans1 --> p1
        p1 = trans1[i,:]
        # p1 on the y-axis --> p1_y
        for j in range(M):
            y_id = poi_dic[j][2]
            p1_y[y_id] += p1[j]
        # p1 on the x-axis --> p1_x
        for j in range(M):
            x_id = poi_dic[j][3]
            p1_x[x_id] += p1[j]

        # i-th row (conditional probability from the i-th POI) in trans2 --> p2
        p2 = trans2[i,:]
        # p2 on the y-axis --> p2_y
        for j in range(M):
            y_id = poi_dic[j][2]
            p2_y[y_id] += p2[j]
        # p2 on the x-axis --> p2_x
        for j in range(M):
            x_id = poi_dic[j][3]
            p2_x[x_id] += p2[j]

        # EMD between p1_y & p2_y --> avg_emd_y
        emd_y = wasserstein_distance(y_axis_ids, y_axis_ids, p1_y, p2_y)
        avg_emd_y += emd_y

        # EMD between p1_x & p2_x --> avg_emd_x
        emd_x = wasserstein_distance(x_axis_ids, x_axis_ids, p1_x, p2_x)
        avg_emd_x += emd_x

    # Calculate the average EMD
    avg_emd_y /= M
    avg_emd_x /= M

    return avg_emd_y, avg_emd_x

########### Calculate the l1 & l2 losses using transition matrices ############
# [input1]: trans1 (M x M matrix)
# [input2]: trans2 (M x M matrix)
# [input3]: M -- Number of POIs
# [output1]: l1_loss
# [output2]: l2_loss
def CalcL1L2TransMat(trans1, trans2, M):
    # Initializaion
    l1_loss = 0.0
    l2_loss = 0.0
    m_num = 0

    # For each POI
    for i in range(M):
        l1_loss_row = 0.0
        l2_loss_row = 0.0
        # i-th row (conditional probability from the i-th POI) in trans1 --> p1
        p1 = trans1[i,:]
        # i-th row (conditional probability from the i-th POI) in trans2 --> p2
        p2 = trans2[i,:]

        if np.sum(p1) > 0:
            # L1 & L2 losses for the row
            for j in range(M):
                l1_loss_row += np.abs(p1[j] - p2[j])
                l2_loss_row += (p1[j] - p2[j])**2

            # Update the average L1 & L2 losses
            l1_loss += l1_loss_row
            l2_loss += l2_loss_row
            m_num += 1

    l1_loss /= m_num
    l2_loss /= m_num

    return l1_loss, l2_loss

#################################### Main #####################################
# Replace XX with City
TUserIndexFile = TUserIndexFile.replace("XX", City)
EUserIndexFile = EUserIndexFile.replace("XX", City)
POIFile = POIFile.replace("XX", City)
POIIndexFile = POIIndexFile.replace("XX", City)
TrainTraceFile = TrainTraceFile.replace("XX", City)
TestTraceFile = TestTraceFile.replace("XX", City)
ResFile = ResFile.replace("XX", City)

# Number of training users --> N
N = len(open(TUserIndexFile).readlines()) - 1
# Number of testing users --> N2
N2 = len(open(EUserIndexFile).readlines()) - 1
# Number of POIs --> M
M = len(open(POIIndexFile).readlines()) - 1
# Number of time slots --> T
if TimeType == 1:
    T = 30
elif TimeType == 2:
    T = 12
else:
    print("Wrong TimeType.\n")
    sys.exit(-1)

# Read a training/testing trace file for the MAP re-identification attack (DataSet = PF)
if DataSet == "PF":
    ttrans_count, ttrans_prob, tcount_sum = ReadTrainTraceFile()
    etrans_count, etrans_prob, ecount_sum = ReadTestTraceFile()

# Read POI files
poi_dic = ReadPOI()

# Read model parameters
ParamA_bin = np.zeros((N,K))
if os.path.exists(ModelParameterDir):
    ParamA, ParamB, ParamC, ParamD = ReadModelParameters()

    # Normalize model parameters
    for k in range(K):
        l2_norm = np.linalg.norm(ParamA[:,k])
        ParamA[:,k] /= l2_norm
        l2_norm = np.linalg.norm(ParamB[:,k])
        ParamB[:,k] /= l2_norm
        l2_norm = np.linalg.norm(ParamC[:,k])
        ParamC[:,k] /= l2_norm
        l2_norm = np.linalg.norm(ParamD[:,k])
        ParamD[:,k] /= l2_norm

    for k in range(K):
        # Set the 90th percentile of ParamA as ParamAThr
        ParamAThr = np.percentile(ParamA[:,k], 90)
        # Binarize model parameter A using ParamAThr
        ParamA_bin[:,k] = ParamA[:,k] > ParamAThr

# Read training traces
ttrain_dist, ttrain_count, atrain_trans, vf_train_dist, ktrain_dist, kttrain_dist = ReadRealTraces(0, N, M, T, ParamA_bin, TrainTraceFile)

# Read testing traces
ttest_dist, ttest_count, atest_trans, vf_test_dist, ktest_dist, kttest_dist = ReadRealTraces(N, N2, M, T, ParamA_bin, TestTraceFile)

# Set vf_exist
vf_exist = np.ones(M)
for i in range(M):
    if np.sum(vf_train_dist[i]) == 0 or np.sum(vf_test_dist[i]) == 0:
        vf_exist[i] = 0

SynTraceFileLst = glob.glob(SynTraceFileAst)

f = open(ResFile, "w")
print("tracedir, tracefile, reid_rate, MIA_acc, MIA_adv, -, TP-TV_syn, TP-TV_tes, TP-TV_uni, -, TP-TV-Top50_syn, TP-TV-Top50_tes, TP-TV-Top50_uni, -, TP-SWD_syn, TP-SWD_tes, TP-SWD_uni, -, VF-TV_syn, VF-TV_tes, VF-TV_uni, -, TM-TV_syn, TM-TV_tes, TM-TV_uni, -, TM-SWD_syn, TM-SWD_tes, TM-SWD_uni", file=f)

######################### Utiility of the benchmark  ##########################
################### Time-specific Geo-distribution ####################
# Uniform distribution --> uni_dist
tuni_dist = np.full((T, M), 1.0/float(M))
tones = np.ones(T)

# Calculate the SWD between ttrain_dist & ttest_dist
ttes_swd = CalcSWD(ttrain_dist, ttest_dist, T, M)
# Calculate the SWD between ttrain_dist & tuni_dist
tuni_swd = CalcSWD(ttrain_dist, tuni_dist, T, M)
    
# Calculate the l1 & l2 losses between ttrain_dist & ttest_dist
ttes_l1_loss, ttes_l2_loss = CalcL1L2(ttrain_dist, ttest_dist, tones, T, M, M)
# Calculate the l1 & l2 losses between ttrain_dist & tuni_dist
tuni_l1_loss, tuni_l2_loss = CalcL1L2(ttrain_dist, tuni_dist, tones, T, M, M)

####################### Time-specific Top-L POIs ######################
# Calculate the l1 & l2 losses between ttrain_dist & ttest_dist
ttesl_l1_loss, ttesl_l2_loss = CalcL1L2(ttrain_dist, ttest_dist, tones, T, M, L1)
# Calculate the l1 & l2 losses between ttrain_dist & tuni_dist
tunil_l1_loss, tunil_l2_loss = CalcL1L2(ttrain_dist, tuni_dist, tones, T, M, L1)

######### Visit-fraction distribution [Ye+,KDD11][Do+,TMC13] ##########
vtes_l1 = 0
vuni_l1 = 0
if DataSet == "FS":
    # Uniform distribution --> vf_uni_dist
    #vf_uni_dist = np.full((M, B), 1.0/float(B))
    vf_uni_dist = np.zeros((M, B))
    vf_bin = math.ceil(1.0*B/M) - 1
    for i in range(M):
        vf_uni_dist[i,vf_bin] = 1
    
    # Calculate the l1-loss between vf_train_dist & vf_test_dist
    vtes_l1 = CalcL1VfDist(vf_train_dist, vf_test_dist, vf_exist, M, B)
    # Calculate the l1-loss between vf_train_dist & vf_uni_dist
    vuni_l1 = CalcL1VfDist(vf_train_dist, vf_uni_dist, vf_exist, M, B)

########################## Mobility features ##########################
# Uniform transition matrix --> uni_trans
auni_trans = np.full((M, M), 1.0/float(M))

# Calculate the average SWD between atrain_trans & atest_trans
ates_trans_swd = CalcSWDTransMat(atrain_trans, atest_trans, poi_dic, M)
# Calculate the l1 losss between atrain_trans & atest_trans
ates_trans_l1, ates_trans_l2 = CalcL1L2TransMat(atrain_trans, atest_trans, M)

# Calculate the average EMD on the y/x-axis between atrain_trans & auni_trans
auni_trans_swd = CalcSWDTransMat(atrain_trans, auni_trans, poi_dic, M)
# Calculate the l1 losss between atrain_trans & atrain_trans
auni_trans_l1, auni_trans_l2 = CalcL1L2TransMat(atrain_trans, auni_trans, M)

# For each synthesized trace file
for SynTraceFile in SynTraceFileLst:
    SynTraceFile = SynTraceFile.replace("\\", "/")
    print("Evaluating", os.path.split(SynTraceFile)[0].split("/")[-1] + "/" + os.path.split(SynTraceFile)[1])

    if DataSet == "PF":
        # MAP (Maximum a Posteriori) re-identification attack --> reid_res
        log_post, reid_res = MAPReidentify(ttrans_prob, tcount_sum, SynTraceFile)
        reid_num = 0
        for i in range(N*TraceNum):
            if reid_res[i] == int(i / TraceNum):
                reid_num += 1
        reid_rate = reid_num / (N*TraceNum)

        # Likelihood-ratio-based MIA (Membership Inference Attack) --> mia_res
        llr_per_trace, MIA_thr, MIA_true_pos, MIA_true_neg, MIA_max_acc, MIA_max_adv = LRMIA(ttrans_prob, etrans_prob, SynTraceFile)
#        # Output the detailed results of MIA
#        outfile = DataDir + "utilpriv_MIA_" + os.path.split(SynTraceFile)[0].split("/")[-1] + "_" + os.path.split(SynTraceFile)[1]
#        f2 = open(outfile, "w")
#        print("thr, #true_pos, #true_neg, accuracy, advantage", file=f2)
#        writer = csv.writer(f2, lineterminator="\n")
#        for i in range(1000):
#            s = [MIA_thr[i], MIA_true_pos[i], MIA_true_neg[i], 
#                 (MIA_true_pos[i]+MIA_true_neg[i])/(TraceNum*(N+N2)),
#                 MIA_true_pos[i]/(TraceNum*N) - 1 + MIA_true_neg[i]/(TraceNum*N2)]            
#            writer.writerow(s)
#        f2.close()
    else:
        reid_rate = 0
        MIA_max_acc = 0
        MIA_max_adv = 0

    # Initialization
    tsyn_swd_avg = 0
    tsyn_l1_loss_avg = 0
    tsyn_l2_loss_avg = 0

    tsynl_l1_loss_avg = 0
    tsynl_l2_loss_avg = 0

    vsyn_l1_avg = 0

    ksyn_l1_loss = np.zeros(K)
    ksyn_l2_loss = np.zeros(K)

#    asyn_trans_emd_y_avg = 0
#    asyn_trans_emd_x_avg = 0
    asyn_trans_swd_avg = 0
    asyn_trans_l1_avg = 0

    # PD test result file --> PDTestResFile
    if DataSet == "FS" and PDTest == 1 and (SynAlg == "PPMTF" or SynAlg == "PPITF"):
        PDTestResFile = DataDir + os.path.split(SynTraceFile)[0].split("/")[-1] + "/" + "pdtest_res"
    else:
        PDTestResFile = "none"

    # For each trace no.
    for trace_no in range(TraceNum):
        # Read synthesized traces
        tsyn_dist, asyn_trans, pass_test, vf_syn_dist, ksyn_dist, ktsyn_dist = ReadSynTraces(N, M, T, ParamA_bin, SynTraceFile, PDTestResFile, Reqk, trace_no)

        ################### Time-specific Geo-distribution ####################
        # Calculate the SWD between ttrain_dist & ttest_dist
        tsyn_swd = CalcSWD(ttrain_dist, tsyn_dist, T, M)
        tsyn_swd_avg += tsyn_swd

        # Calculate the l1 & l2 losses between ttrain_dist & tsyn_dist
        tsyn_l1_loss, tsyn_l2_loss = CalcL1L2(ttrain_dist, tsyn_dist, tones, T, M, M)
        tsyn_l1_loss_avg += tsyn_l1_loss
        tsyn_l2_loss_avg += tsyn_l2_loss

        ####################### Time-specific Top-L POIs ######################
        # Calculate the l1 & l2 losses between ttrain_dist & tsyn_dist
        tsynl_l1_loss, tsynl_l2_loss = CalcL1L2(ttrain_dist, tsyn_dist, tones, T, M, L1)
        tsynl_l1_loss_avg += tsynl_l1_loss
        tsynl_l2_loss_avg += tsynl_l2_loss
    
        ######### Visit-fraction distribution [Ye+,KDD11][Do+,TMC13] ##########
        if DataSet == "FS":
            # Calculate the average EMD between vf_train_dist & vf_syn_dist
            vsyn_l1 = CalcL1VfDist(vf_train_dist, vf_syn_dist, vf_exist, M, B)
            vsyn_l1_avg += vsyn_l1
    
        ########################## Mobility features ##########################
        # Calculate the average SWD between atrain_trans & asyn_trans
        asyn_trans_swd = CalcSWDTransMat(atrain_trans, asyn_trans, poi_dic, M)
        asyn_trans_swd_avg += asyn_trans_swd
        # Calculate the l1 losss between atrain_trans & asyn_trans
        asyn_trans_l1, asyn_trans_l2 = CalcL1L2TransMat(atrain_trans, asyn_trans, M)
        asyn_trans_l1_avg += asyn_trans_l1

    # Normalization
    tsyn_swd_avg /= TraceNum
    tsyn_l1_loss_avg /= TraceNum
    tsyn_l2_loss_avg /= TraceNum

    tsynl_l1_loss_avg /= TraceNum
    tsynl_l2_loss_avg /= TraceNum

    vsyn_l1_avg /= TraceNum

    asyn_trans_swd_avg /= TraceNum
    asyn_trans_l1_avg /= TraceNum

    # Output the results
    writer = csv.writer(f, lineterminator="\n")
    if DataSet == "PF":
        s = [os.path.split(SynTraceFile)[0].split("/")[-1], os.path.split(SynTraceFile)[1], reid_rate, MIA_max_acc, MIA_max_adv, "-", 
             tsyn_l1_loss_avg/2.0, ttes_l1_loss/2.0, tuni_l1_loss/2.0, "-", 
             tsynl_l1_loss_avg/2.0, ttesl_l1_loss/2.0, tunil_l1_loss/2.0, "-", 
             tsyn_swd_avg, ttes_swd, tuni_swd, "-", 
             "-", "-", "-", "-", 
             asyn_trans_l1_avg/2.0, ates_trans_l1/2.0, auni_trans_l1/2.0, "-", 
             asyn_trans_swd_avg, ates_trans_swd, auni_trans_swd]
    else:
        s = [os.path.split(SynTraceFile)[0].split("/")[-1], os.path.split(SynTraceFile)[1], "-", "-", "-", "-", 
             tsyn_l1_loss_avg/2.0, ttes_l1_loss/2.0, tuni_l1_loss/2.0, "-", 
             tsynl_l1_loss_avg/2.0, ttesl_l1_loss/2.0, tunil_l1_loss/2.0, "-", 
             tsyn_swd_avg, ttes_swd, tuni_swd, "-", 
             vsyn_l1_avg/2.0, vtes_l1/2.0, vuni_l1/2.0, "-", 
             asyn_trans_l1_avg/2.0, ates_trans_l1/2.0, auni_trans_l1/2.0, "-", 
             asyn_trans_swd_avg, ates_trans_swd, auni_trans_swd]
    writer.writerow(s)

f.close()
