#!/usr/bin/env python3
import numpy as np
import csv
import sys

################################# Parameters ##################################
#sys.argv = ["SynData_PPMTF.py", "SmplA1"]
if len(sys.argv) < 2:
    print("Usage:",sys.argv[0],"[ParamA]" )
    sys.exit(0)

# City
City = "TK"
#City = "OS"
# Training user index file (input)
TUserIndexFile = "data/tuserindex_XX.csv"
# POI index file (input)
POIIndexFile = "data/POIindex_XX.csv"
# Training transition tensor file (input)
TrainTransTensorFile = "data/traintranstensor_XX.csv"
# Training visit tensor file (input)
TrainVisitTensorFile = "data/trainvisittensor_XX.csv"
# Prefix of the model parameter file (input)
ModelParameterFile = "data/models_syntraces_XX/modelparameter"
# Prefix of the synthesized trace file (output)
SynTraceFile = "data/models_syntraces_XX/syntraces"
# Name of the model parameter A
ParamA = sys.argv[1]
#ParamA = "SmplA"

# Number of time periods
T = 12
# Number of columns in model parameters (A, B, C)
K = 32
#K = 16
# Number of iterations in Gibbs sampling
ItrNum = 100
#ItrNum = 10
# Threshold for a visit count
#VisThr = 0.5
VisThr = 0
# Minimum value of a visit count
VisitDelta = 0.00000001
# Minimum value of a transition count
TransDelta = 0.00000001
# Read trans from TrainTransTensorFile (1:yes, 0:no)
ReadTrans = 1
#ReadTrans = 0
# Read visits from TrainVisitTensorFile (1:yes, 0:no)
ReadVisit = 1
#ReadVisit = 0

# Number of traces per user
#TraceNum = 1
TraceNum = 60
# Number of time instants per time period
#TimInsNum = 1
TimInsNum = 12

# Increase visit-counts (normalized to [0,1]) for a specific location (home) at 6-7h & 19-24h (time_poi_dist) by Gamma
Gamma = 20

# Number of synthesized users
SynN = 2000

########################### Read model parameters #############################
# [output1]: A (N x K matrix)
# [output2]: B (M x K matrix)
# [output3]: C (M x K matrix)
# [output4]: D (T x K matrix)
def ReadModelParameters():
    # Read model parameter A
    infile = ModelParameterFile + "_K" + str(K) + "_Itr" + str(ItrNum) + "_" + ParamA + ".csv"
    f = open(infile, "r")
    A = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter B
    infile = ModelParameterFile + "_K" + str(K) + "_Itr" + str(ItrNum) + "_B.csv"
    f = open(infile, "r")
    B = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter C
    infile = ModelParameterFile + "_K" + str(K) + "_Itr" + str(ItrNum) + "_C.csv"
    f = open(infile, "r")
    C = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read model parameter D
    infile = ModelParameterFile + "_K" + str(K) + "_Itr" + str(ItrNum) + "_D.csv"
    f = open(infile, "r")
    D = np.loadtxt(infile, delimiter=",")
    f.close()

    return A, B, C, D

############################## Synthesize traces ##############################
# [input1]: A (N x K matrix)
# [input2]: B (M x K matrix)
# [input3]: D (T x K matrix)
# [input4]: N -- Number of users
# [input5]: M -- Number of POIs
# [input6]: T -- Number of time periods
# [input7]: poi_dic ({poi_index: category})
def SynTraces(A, B, D, N, M, T, poi_dic):
    # Initialization
    ab = np.zeros(M)
    ad = np.zeros(M)

    # Output header information
    outfile = SynTraceFile + "_K" + str(K) + "_Itr" + str(ItrNum) + "_" + ParamA + ".csv"
    f = open(outfile, "w")
    print("user,trace_no,time_period,time_instant,poi_index,category", file=f)
    writer = csv.writer(f, lineterminator="\n")

    # Read transitions from TrainTransTensorFile --> trans
    if ReadTrans == 1:
        trans = np.zeros((M, M))
        g = open(TrainTransTensorFile, "r")
        reader = csv.reader(g)
        next(reader)
        for lst in reader:
            poi_index_from = int(lst[1])
            poi_index_to = int(lst[2])
            trans[poi_index_from,poi_index_to] = 1      
        g.close()

    # Read visits from TrainVisitTensorFile --> visit
    if ReadVisit == 1:
        visit = np.zeros((T, M))
        g = open(TrainVisitTensorFile, "r")
        reader = csv.reader(g)
        next(reader)
        for lst in reader:
            poi_index_from = int(lst[1])
            time_id = int(lst[2])
            visit[time_id,poi_index_from] = 1       
        g.close()

    HomeFileName = "home_" + ParamA + ".csv"
    g = open(HomeFileName, "w")
    # For each user
    for n in range(SynN):
        # Initialization
        time_poi_dist = np.zeros((T, M))
        time_poi_dist_sum = np.zeros(T)
        prop_mat = np.zeros((M, M))
        trans_vec = np.zeros(M)

        ################### Calculate the POI distributions ###################
        for t in range(T):
            ad = A[n, :] * D[t, :]
            for i in range(M):
                # Elements in a sampled visit tensor --> time_poi_dist
                time_poi_dist[t,i] = np.sum(ad * B[i, :])
                # Assign VisitDelta for an element whose value is less than VisThr
                if time_poi_dist[t,i] < VisThr:
                    time_poi_dist[t,i] = VisitDelta

                # Assign VisitDelta if there is no visits for time t & user i
                if ReadVisit == 1 and visit[t,i] == 0:
                    time_poi_dist[t,i] = VisitDelta

        # Normalize time_poi_dist (this is necessary for randomly sampling home_loc)
        for t in range(T):
            time_poi_dist_sum[t] = np.sum(time_poi_dist[t])
            if time_poi_dist_sum[t] > 0:
                time_poi_dist[t] /= time_poi_dist_sum[t]
            else:
                print("Error: All probabilities are 0 for user", n, "and time", t)
                sys.exit(-1)

        # Randomly sample home from the POI distribution at 6h --> home_loc
        rnd = np.random.rand()
        prob_sum = 0
        for i in range(M):
            prob_sum += time_poi_dist[0,i]
            if prob_sum >= rnd:
                break
        home_loc = i
#        print(home_loc)
        print(home_loc, file=g)
        # Increase visit-counts for home_loc at 6-7h & 18-21h (time_poi_dist) by Gamma
        for t in range(2):
            time_poi_dist[t,home_loc] += Gamma
        for t in range(T-2,T):
            time_poi_dist[t,home_loc] += Gamma

        # Normalize time_poi_dist at 6-7h & 18-21h (again)
        for t in range(2):
            time_poi_dist_sum[t] = np.sum(time_poi_dist[t])
            if time_poi_dist_sum[t] > 0:
                time_poi_dist[t] /= time_poi_dist_sum[t]
            else:
                print("Error: All probabilities are 0 for user", n, "and time", t)
                sys.exit(-1)
        for t in range(T-3,T):
            time_poi_dist_sum[t] = np.sum(time_poi_dist[t])
            if time_poi_dist_sum[t] > 0:
                time_poi_dist[t] /= time_poi_dist_sum[t]
            else:
                print("Error: All probabilities are 0 for user", n, "and time", t)
                sys.exit(-1)

        #################### Calculate the proposal matrix ####################
        for i in range(M):
            ab = A[n, :] * B[i, :]
            # Elements in a sampled transition tensor (assign TransDelta for a small transition count) --> prop_mat
            for j in range(M):
                prop_mat[i,j] = max(np.sum(ab * C[j, :]), TransDelta)
                # Assign TransDelta if there is no transitions between i and j
                if ReadTrans == 1 and trans[i,j] == 0:
                    prop_mat[i,j] = TransDelta

            # Normalize prop_mat
            row_sum = np.sum(prop_mat[i])
            prop_mat[i] /= row_sum

        ########################## Synthesize traces ##########################
        poi_index_pre = 0
        # For each trace
        for trace_no in range(TraceNum):
            # For each time period
            for t in range(T):
                # For each time instant
                for ins in range(TimInsNum):
                    # Initial time period and initial event
                    if t == 0 and ins == 0:
                        # Randomly sample POI from the POI distribution
                        rnd = np.random.rand()
                        prob_sum = 0
                        for i in range(M):
                            prob_sum += time_poi_dist[t,i]
                            if prob_sum >= rnd:
                                break
                        poi_index = i
                    else:
                        ##### Transform poi_index_pre into poi_index via MH (Metropolis-Hastings) ######
                        # Calculate the transition vector --> trans_vec
                        trans_vec[poi_index_pre] = 0
                        for j in range(M):
                            if poi_index_pre != j:
                                alpha = (time_poi_dist[t][j] * prop_mat[j,poi_index_pre]) / (time_poi_dist[t][poi_index_pre] * prop_mat[poi_index_pre,j])
                                trans_vec[j] = prop_mat[poi_index_pre,j] * min(1, alpha)
                        row_sum = np.sum(trans_vec)
                        trans_vec[poi_index_pre] = 1 - row_sum

                        # Transform poi_index_pre into poi_index via trans_vec
                        rnd = np.random.rand()
                        prob_sum = 0
                        for j in range(M):
                            prob_sum += trans_vec[j]
                            if prob_sum >= rnd:
                                break
                        poi_index = j

                    # Output an initial location ([user, trace_no, time_period, time_instant, poi_index, category])
                    s = [n, trace_no, t, ins, poi_index, poi_dic[poi_index]]
                    writer.writerow(s)

                    # Save the previous poi_index
                    poi_index_pre = poi_index
    f.close()
    g.close()

#################################### Main #####################################
# Fix a seed
#np.random.seed(1)
# Fix a seed using a random number in [0,2^32-1]
#np.random.seed(819081307) # Preliminary
np.random.seed(538173108) # Final (TK)

# Replace XX with City
TUserIndexFile = TUserIndexFile.replace("XX", City)
POIIndexFile = POIIndexFile.replace("XX", City)
TrainTransTensorFile = TrainTransTensorFile.replace("XX", City)
TrainVisitTensorFile = TrainVisitTensorFile.replace("XX", City)
ModelParameterFile = ModelParameterFile.replace("XX", City)
SynTraceFile = SynTraceFile.replace("XX", City)

# Number of training users --> N
N = len(open(TUserIndexFile).readlines()) - 1
# Number of POIs --> M
M = len(open(POIIndexFile).readlines()) - 1

# Read the POI index file --> poi_dic ({poi_index: category})
poi_dic = {}
f = open(POIIndexFile, "r")
reader = csv.reader(f)
next(reader)
for lst in reader:
    poi_dic[int(lst[1])] = lst[2]

# Read model parameters
A, B, C, D = ReadModelParameters()

# Synthesize traces
SynTraces(A, B, D, N, M, T, poi_dic)
