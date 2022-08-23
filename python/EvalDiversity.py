#!/usr/bin/env python3
import numpy as np
import random
from scipy.stats import wasserstein_distance
from scipy import stats
import csv
import sys

################################# Parameters ##################################
#sys.argv = ["EvalUtilPriv_Diversity.py", "FS", "IS", "SmplA", 20, 1000]
#sys.argv = ["EvalUtilPriv_Diversity.py", "FS", "IS", "A", 20, 1000]

if len(sys.argv) < 4:
    print("Usage:",sys.argv[0],"[Dataset] [City] [ParamA (A/SmplA)] ([Trace Num (default: 10)] [Selected User Num (default: -1)])")
    sys.exit(0)

# Dataset (PF/FS)
DataSet = sys.argv[1]
# City
City = sys.argv[2]

# ParamA
ParamA = sys.argv[3]

# Trace Num
TraceNum = 10
if len(sys.argv) >= 5:
    TraceNum = int(sys.argv[4])

# Selected User Num (-1: all)
SUserNum = -1
if len(sys.argv) >= 6:
    SUserNum = int(sys.argv[5])
 
# Data directory
DataDir = "../data/" + DataSet + "/"
# Training user index file (input)
TUserIndexFile = DataDir + "tuserindex_XX.csv"
# POI file (input)
POIFile = DataDir + "POI_XX.csv"
# POI index file (input)
POIIndexFile = DataDir + "POIindex_XX.csv"
# Training trace file (input)
TrainTraceFile = DataDir + "traintraces_XX.csv"
# Result file (output)
ResFile = DataDir + "div_" + City + "_" + ParamA + "_tn" + str(TraceNum) + "_sn" + str(SUserNum) + ".csv"

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

# Synthetic trace file (input)
if ParamA == "SmplA":
    SynTraceFile1 = DataDir + "PPMTF_XX_alp200_mnt100_mnv100/syntraces_Itr100_SmplA1.csv"
    SynTraceFile2 = DataDir + "PPMTF_XX_alp200_mnt100_mnv100/syntraces_Itr100_SmplA2.csv"
    SynTraceFile2 = SynTraceFile2.replace("XX", City)
elif ParamA == "A":
    SynTraceFile1 = DataDir + "PPMTF_XX_alp200_mnt100_mnv100/syntraces_Itr100_A.csv"
else:
    print("Wrong ParamA")
    sys.exit(-1)

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

############################## Read real traces ###############################
# [input1]: suser_index -- Selected user index
# [input2]: auser_index -- Another user index
# [output1]: ureal_dist_u1 (M-dim vector)
# [output2]: ureal_dist_u2 (M-dim vector)
# [output3]: ureal_trans_u1 (M x M matrix)
# [output4]: ureal_trans_u2 (M x M matrix)
def ReadRealTraces(suser_index, auser_index):
    ureal_dist_u1 = np.zeros(M)
    ureal_dist_u2 = np.zeros(M)
    ureal_trans_u1 = np.zeros((M, M))
    ureal_trans_u2 = np.zeros((M, M))
    user_index_prev = -1
    poi_index_prev = 0

    # Read a real trace file
    f = open(TrainTraceFile, "r")
    reader = csv.reader(f)
    next(reader)
    for lst in reader:
        # Read only suser_index or auser_index
        user_index = int(lst[0])
        if user_index != suser_index and user_index != auser_index:
            continue

        poi_index = int(lst[1])
        # 1st user
        if user_index == suser_index:
            # Update the user-specific real distribution --> ureal_dist_u1
            ureal_dist_u1[poi_index] += 1
            # Update the user-specific real transition matrix --> ureal_trans_u1
            if user_index == user_index_prev:
                ureal_trans_u1[poi_index_prev, poi_index] += 1
        # 2nd user
        elif user_index == auser_index:
            # Update the user-specific real distribution --> ureal_dist_u2
            ureal_dist_u2[poi_index] += 1
            # Update the user-specific real transition matrix --> ureal_trans_u2
            if user_index == user_index_prev:
                ureal_trans_u2[poi_index_prev, poi_index] += 1

        user_index_prev = user_index
        poi_index_prev = poi_index
    f.close()

    # Normalize ureal_dist_u1
    if np.sum(ureal_dist_u1) > 0:
        ureal_dist_u1 /= np.sum(ureal_dist_u1)
    else:
        ureal_dist_u1 = np.full(M, 1.0/float(M))
    # Normalize ureal_dist_u2
    if np.sum(ureal_dist_u2) > 0:
        ureal_dist_u2 /= np.sum(ureal_dist_u2)
    else:
        ureal_dist_u2 = np.full(M, 1.0/float(M))

    # Normalize ureal_trans_u1
    for i in range(M):
        if np.sum(ureal_trans_u1[i]) > 0:
            ureal_trans_u1[i] /= np.sum(ureal_trans_u1[i])
        else:
            ureal_trans_u1[i,i] = 1.0
    # Normalize ureal_trans_u2
    for i in range(M):
        if np.sum(ureal_trans_u2[i]) > 0:
            ureal_trans_u2[i] /= np.sum(ureal_trans_u2[i])
        else:
            ureal_trans_u2[i,i] = 1.0

    return ureal_dist_u1, ureal_dist_u2, ureal_trans_u1, ureal_trans_u2

############################## Read real traces ###############################
# [input1]: suser_index -- Selected user index
# [input2]: auser_index -- Another user index
# [output1]: usyn_dist_t1u1 (M-dim vector)
# [output2]: usyn_dist_t1u2 (M-dim vector)
# [output3]: usyn_dist_t2u1 (M-dim vector)
# [output4]: usyn_trans_t1u1 (M x M matrix)
# [output5]: usyn_trans_t1u2 (M x M matrix)
# [output6]: usyn_trans_t2u1 (M x M matrix)
def ReadSynTraces(suser_index, auser_index):
    usyn_dist_t1u1 = np.zeros(M)
    usyn_dist_t1u2 = np.zeros(M)
    usyn_dist_t2u1 = np.zeros(M)
    usyn_trans_t1u1 = np.zeros((M, M))
    usyn_trans_t1u2 = np.zeros((M, M))
    usyn_trans_t2u1 = np.zeros((M, M))
    user_index_prev = -1
    trace_no_prev = -1
    poi_index_prev = 0

    if ParamA == "SmplA":
        # Read a synthetic trace file (1st team)
        f = open(SynTraceFile1, "r")
        reader = csv.reader(f)
        next(reader)
        for lst in reader:
            # Read only suser_index or auser_index
            user_index = int(lst[0])
            if user_index != suser_index and user_index != auser_index:
                continue

            trace_no = int(lst[1])
            poi_index = int(lst[4])
            # 1st team, 1st user
            if user_index == suser_index:
                # Update the user-specific synthetic distribution --> usyn_dist_t1u1
                usyn_dist_t1u1[poi_index] += 1
                # Update the user-specific real transition matrix --> usyn_trans_t1u1
                if user_index == user_index_prev and trace_no == trace_no_prev:
                    usyn_trans_t1u1[poi_index_prev, poi_index] += 1
            # 1st team, 2nd user
            elif user_index == auser_index:
                # Update the user-specific synthetic distribution --> usyn_dist_t1u2
                usyn_dist_t1u2[poi_index] += 1
                # Update the user-specific real transition matrix --> usyn_trans_t1u2
                if user_index == user_index_prev and trace_no == trace_no_prev:
                    usyn_trans_t1u2[poi_index_prev, poi_index] += 1

            user_index_prev = user_index
            trace_no_prev = trace_no
            poi_index_prev = poi_index
        f.close()

        user_index_prev = -1
        trace_no_prev = -1
        poi_index_prev = 0

        # Read a synthetic trace file (2nd team)
        f = open(SynTraceFile2, "r")
        reader = csv.reader(f)
        next(reader)
        for lst in reader:
            # Read only suser_index
            user_index = int(lst[0])
            if user_index != suser_index:
                continue

            trace_no = int(lst[1])
            poi_index = int(lst[4])
            # 2nd team, 1st user
            if user_index == suser_index:
                # Update the user-specific synthetic distribution --> usyn_dist_t2u1
                usyn_dist_t2u1[poi_index] += 1
                # Update the user-specific real transition matrix --> usyn_trans_t2u1
                if user_index == user_index_prev and trace_no == trace_no_prev:
                    usyn_trans_t2u1[poi_index_prev, poi_index] += 1

            user_index_prev = user_index
            trace_no_prev = trace_no
            poi_index_prev = poi_index
        f.close()
    else:
        # Read a synthetic trace file
        f = open(SynTraceFile1, "r")
        reader = csv.reader(f)
        next(reader)
        for lst in reader:
            # Read only suser_index or auser_index
            user_index = int(lst[0])
            if user_index != suser_index and user_index != auser_index:
                continue

            trace_no = int(lst[1])
            poi_index = int(lst[4])
            # 1st team, 1st user
            if trace_no < TraceNum and user_index == suser_index:
                # Update the user-specific synthetic distribution --> usyn_dist_t1u1
                usyn_dist_t1u1[poi_index] += 1
                # Update the user-specific real transition matrix --> usyn_trans_t1u1
                if user_index == user_index_prev and trace_no == trace_no_prev:
                    usyn_trans_t1u1[poi_index_prev, poi_index] += 1
            # 1st team, 2nd user
            elif trace_no < TraceNum and user_index == auser_index:
                # Update the user-specific synthetic distribution --> usyn_dist_t1u2
                usyn_dist_t1u2[poi_index] += 1
                # Update the user-specific real transition matrix --> usyn_trans_t1u2
                if user_index == user_index_prev and trace_no == trace_no_prev:
                    usyn_trans_t1u2[poi_index_prev, poi_index] += 1
            # 2nd team, 1st user
            elif trace_no >= TraceNum and user_index == suser_index:
                # Update the user-specific synthetic distribution --> usyn_dist_t2u1
                usyn_dist_t2u1[poi_index] += 1
                # Update the user-specific real transition matrix --> usyn_trans_t2u1
                if user_index == user_index_prev and trace_no == trace_no_prev:
                    usyn_trans_t2u1[poi_index_prev, poi_index] += 1

            user_index_prev = user_index
            trace_no_prev = trace_no
            poi_index_prev = poi_index
        f.close()

    # Normalize usyn_dist_t1u1
    if np.sum(usyn_dist_t1u1) > 0:
        usyn_dist_t1u1 /= np.sum(usyn_dist_t1u1)
    else:
        usyn_dist_t1u1 = np.full(M, 1.0/float(M))
    # Normalize usyn_dist_t1u2
    if np.sum(usyn_dist_t1u2) > 0:
        usyn_dist_t1u2 /= np.sum(usyn_dist_t1u2)
    else:
        usyn_dist_t1u2 = np.full(M, 1.0/float(M))
    # Normalize usyn_dist_t2u1
    if np.sum(usyn_dist_t2u1) > 0:
        usyn_dist_t2u1 /= np.sum(usyn_dist_t2u1)
    else:
        usyn_dist_t2u1 = np.full(M, 1.0/float(M))

    # Normalize usyn_trans_t1u1
    for i in range(M):
        if np.sum(usyn_trans_t1u1[i]) > 0:
            usyn_trans_t1u1[i] /= np.sum(usyn_trans_t1u1[i])
        else:
            usyn_trans_t1u1[i,i] = 1.0
    # Normalize usyn_trans_t1u2
    for i in range(M):
        if np.sum(usyn_trans_t1u2[i]) > 0:
            usyn_trans_t1u2[i] /= np.sum(usyn_trans_t1u2[i])
        else:
            usyn_trans_t1u2[i,i] = 1.0
    # Normalize usyn_trans_t2u1
    for i in range(M):
        if np.sum(usyn_trans_t2u1[i]) > 0:
            usyn_trans_t2u1[i] /= np.sum(usyn_trans_t2u1[i])
        else:
            usyn_trans_t2u1[i,i] = 1.0

    return usyn_dist_t1u1, usyn_dist_t1u2, usyn_dist_t2u1, usyn_trans_t1u1, usyn_trans_t1u2, usyn_trans_t2u1

#################### Calculate the average l1 & l2 losses #####################
# [input1]: dist1 (Z x M matrix)
# [input2]: dist2 (Z x M matrix)
# [output1]: l1_loss
# [output2]: l2_loss
def CalcL1L2(dist1, dist2):
    l1_loss = 0.0
    l2_loss = 0.0

    # l1 & l2 losses
    # Update the l1-loss & l2-loss
    for i in range(M):
        l1_loss += np.abs(dist1[i] - dist2[i])
        l2_loss += (dist1[i] - dist2[i])**2

    return l1_loss, l2_loss

######### Calculate the EMD on the y/x-axis using transition matrices #########
# [input1]: trans1 (M x M matrix)
# [input2]: trans2 (M x M matrix)
# [input3]: poi_dic ({poi_index: [y, x, y_id, x_id]})
# [output1]: weight_avg_emd -- Weighted average EMD
def CalcEMDTransMat(trans1, trans2, poi_dic):
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
# [output1]: l1_loss
# [output2]: l2_loss
def CalcL1L2TransMat(trans1, trans2):
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

#        print(i, l1_loss_row, l2_loss_row, l1_loss, l2_loss)
    l1_loss /= m_num
    l2_loss /= m_num

    return l1_loss, l2_loss

#################################### Main #####################################
# Fix a seed
#np.random.seed(1)
random.seed(1)

# Replace XX with City
TUserIndexFile = TUserIndexFile.replace("XX", City)
POIFile = POIFile.replace("XX", City)
POIIndexFile = POIIndexFile.replace("XX", City)
TrainTraceFile = TrainTraceFile.replace("XX", City)
SynTraceFile1 = SynTraceFile1.replace("XX", City)

# Number of training users --> N
N = len(open(TUserIndexFile).readlines()) - 1
# Number of POIs --> M
M = len(open(POIIndexFile).readlines()) - 1

if 2*SUserNum > N:
    print("2*SUserNum must be less than or equal to N")
    sys.exit(-1)

# Read POI files
poi_dic = ReadPOI()

# Select users --> user_list
if SUserNum == -1:
    SUserNum = N
    user_list = list(range(N))
    for i in range(N):
        rnd = random.choice(range(N-1))
        if rnd >= i:
            rnd += 1
        user_list.append(rnd)
else:
    user_list = random.sample(range(N), 2*SUserNum)
#user_list.sort()

l1_u1_t1u1_dist = l2_u1_t1u1_dist = np.zeros(SUserNum)
l1_u2_t1u1_dist = l2_u2_t1u1_dist = np.zeros(SUserNum)
l1_t1u1_t1u2_dist = l2_t1u1_t1u2_dist = np.zeros(SUserNum)
l1_t1u1_t2u1_dist = l2_t1u1_t2u1_dist = np.zeros(SUserNum)

emdy_u1_t1u1_trans = emdx_u1_t1u1_trans = np.zeros(SUserNum)
emdy_u2_t1u1_trans = emdx_u2_t1u1_trans = np.zeros(SUserNum)
emdy_t1u1_t1u2_trans = emdx_t1u1_t1u2_trans = np.zeros(SUserNum)
emdy_t1u1_t2u1_trans = emdx_t1u1_t2u1_trans = np.zeros(SUserNum)

l1_u1_t1u1_trans = l2_u1_t1u1_trans = np.zeros(SUserNum)
l1_u2_t1u1_trans = l2_u2_t1u1_trans = np.zeros(SUserNum)
l1_t1u1_t1u2_trans = l2_t1u1_t1u2_trans = np.zeros(SUserNum)
l1_t1u1_t2u1_trans = l2_t1u1_t2u1_trans = np.zeros(SUserNum)

# For each selected user
for i in range(SUserNum):
    print(i)
    # Selected user index --> suser_index
    suser_index = user_list[i]
    # Another user index --> auser_id
    auser_index = user_list[i + SUserNum]

    # Read training traces
    ureal_dist_u1, ureal_dist_u2, ureal_trans_u1, ureal_trans_u2 = ReadRealTraces(suser_index, auser_index)
    # Read synthetic traces
    usyn_dist_t1u1, usyn_dist_t1u2, usyn_dist_t2u1, usyn_trans_t1u1, usyn_trans_t1u2, usyn_trans_t2u1 = ReadSynTraces(suser_index, auser_index)

    # Calculate the l1 & l2 losses between ureal_dist_u1 & usyn_dist_t1u1
    l1_u1_t1u1_dist[i], l2_u1_t1u1_dist[i] = CalcL1L2(ureal_dist_u1, usyn_dist_t1u1)
    # Calculate the l1 & l2 losses between ureal_dist_u2 & usyn_dist_t1u1
    l1_u2_t1u1_dist[i], l2_u2_t1u1_dist[i] = CalcL1L2(ureal_dist_u2, usyn_dist_t1u1)
    # Calculate the l1 & l2 losses between usyn_dist_t1u1 & usyn_dist_t1u2
    l1_t1u1_t1u2_dist[i], l2_t1u1_t1u2_dist[i] = CalcL1L2(usyn_dist_t1u1, usyn_dist_t1u2)
    # Calculate the l1 & l2 losses between usyn_dist_t1u1 & usyn_dist_t2u1
    l1_t1u1_t2u1_dist[i], l2_t1u1_t2u1_dist[i] = CalcL1L2(usyn_dist_t1u1, usyn_dist_t2u1)

    # Calculate the average EMD on the y/x-axis between ureal_trans_u1 & usyn_trans_t1u1
    emdy_u1_t1u1_trans[i], emdx_u1_t1u1_trans[i] = CalcEMDTransMat(ureal_trans_u1, usyn_trans_t1u1, poi_dic)
    # Calculate the average EMD on the y/x-axis between ureal_trans_u2 & usyn_trans_t1u1
    emdy_u2_t1u1_trans[i], emdx_u2_t1u1_trans[i] = CalcEMDTransMat(ureal_trans_u2, usyn_trans_t1u1, poi_dic)
    # Calculate the average EMD on the y/x-axis between usyn_trans_t1u1 & usyn_trans_t1u2
    emdy_t1u1_t1u2_trans[i], emdx_t1u1_t1u2_trans[i] = CalcEMDTransMat(usyn_trans_t1u1, usyn_trans_t1u2, poi_dic)
    # Calculate the average EMD on the y/x-axis between usyn_trans_t1u1 & usyn_trans_t2u1
    emdy_t1u1_t2u1_trans[i], emdx_t1u1_t2u1_trans[i] = CalcEMDTransMat(usyn_trans_t1u1, usyn_trans_t2u1, poi_dic)

    # Calculate the l1 & l2 losses between ureal_trans_u1 & usyn_trans_t1u1
    l1_u1_t1u1_trans[i], l2_u1_t1u1_trans[i] = CalcL1L2TransMat(ureal_trans_u1, usyn_trans_t1u1)
    # Calculate the l1 & l2 losses between ureal_trans_u2 & usyn_trans_t1u1
    l1_u2_t1u1_trans[i], l2_u2_t1u1_trans[i] = CalcL1L2TransMat(ureal_trans_u2, usyn_trans_t1u1)
    # Calculate the l1 & l2 losses between usyn_trans_t1u1 & usyn_trans_t1u2
    l1_t1u1_t1u2_trans[i], l2_t1u1_t1u2_trans[i] = CalcL1L2TransMat(usyn_trans_t1u1, usyn_trans_t1u2)
    # Calculate the l1 & l2 losses between usyn_trans_t1u1 & usyn_trans_t2u1
    l1_t1u1_t2u1_trans[i], l2_t1u1_t2u1_trans[i] = CalcL1L2TransMat(usyn_trans_t1u1, usyn_trans_t2u1)

# Output the results (AVG and STD)
f = open(ResFile, "w")
print("users, TP-TV(AVG), TP-TV(STD), TM-EMD-Y(AVG), TM-EMD-Y(STD), TM-EMD-X(AVG), TM-EMD-X(STD), TM-TV(AVG), TM-TV(STD)", file=f)
writer = csv.writer(f, lineterminator="\n")
s = ["u1-t1u1", np.mean(l1_u1_t1u1_dist), np.std(l1_u1_t1u1_dist), np.mean(emdy_u1_t1u1_trans), np.std(emdy_u1_t1u1_trans), 
     np.mean(emdx_u1_t1u1_trans), np.std(emdx_u1_t1u1_trans), np.mean(l1_u1_t1u1_trans), np.std(l1_u1_t1u1_trans)]
writer.writerow(s)
s = ["u2-t1u1", np.mean(l1_u2_t1u1_dist), np.std(l1_u2_t1u1_dist), np.mean(emdy_u2_t1u1_trans), np.std(emdy_u2_t1u1_trans), 
     np.mean(emdx_u2_t1u1_trans), np.std(emdx_u2_t1u1_trans), np.mean(l1_u2_t1u1_trans), np.std(l1_u2_t1u1_trans)]
writer.writerow(s)
s = ["t1u1-t2u1", np.mean(l1_t1u1_t2u1_dist), np.std(l1_t1u1_t2u1_dist), np.mean(emdy_t1u1_t2u1_trans), np.std(emdy_t1u1_t2u1_trans), 
     np.mean(emdx_t1u1_t2u1_trans), np.std(emdx_t1u1_t2u1_trans), np.mean(l1_t1u1_t2u1_trans), np.std(l1_t1u1_t2u1_trans)]
writer.writerow(s)
s = ["t1u1-t1u2", np.mean(l1_t1u1_t1u2_dist), np.std(l1_t1u1_t1u2_dist), np.mean(emdy_t1u1_t1u2_trans), np.std(emdy_t1u1_t1u2_trans), 
     np.mean(emdx_t1u1_t1u2_trans), np.std(emdx_t1u1_t1u2_trans), np.mean(l1_t1u1_t1u2_trans), np.std(l1_t1u1_t1u2_trans)]
writer.writerow(s)

# Output the results (t-statistics and p-value)
l1dist_u1_tstat, l1dist_u1_pvalue = stats.ttest_rel(l1_u1_t1u1_dist, l1_u2_t1u1_dist)
emdy_u1_tstat, emdy_u1_pvalue = stats.ttest_rel(emdy_u1_t1u1_trans, emdy_u2_t1u1_trans)
emdx_u1_tstat, emdx_u1_pvalue = stats.ttest_rel(emdx_u1_t1u1_trans, emdx_u2_t1u1_trans)
l1trans_u1_tstat, l1trans_u1_pvalue = stats.ttest_rel(l1_u1_t1u1_trans, l1_u2_t1u1_trans)

l1dist_t1u1_tstat, l1dist_t1u1_pvalue = stats.ttest_rel(l1_t1u1_t2u1_dist, l1_t1u1_t1u2_dist)
emdy_t1u1_tstat, emdy_t1u1_pvalue = stats.ttest_rel(emdy_t1u1_t2u1_trans, emdy_t1u1_t1u2_trans)
emdx_t1u1_tstat, emdx_t1u1_pvalue = stats.ttest_rel(emdx_t1u1_t2u1_trans, emdx_t1u1_t1u2_trans)
l1trans_t1u1_tstat, l1trans_t1u1_pvalue = stats.ttest_rel(l1_t1u1_t2u1_trans, l1_t1u1_t1u2_trans)

print("users, TP-TV(t-stat), TP-TV(p-value), TM-EMD-Y(t-stat), TM-EMD-Y(p-value), TM-EMD-X(t-stat), TM-EMD-X(p-value), TM-TV(t-stat), TM-TV(p-value)", file=f)
writer = csv.writer(f, lineterminator="\n")
s = ["u1-t1u1 vs. u2-t1u1", l1dist_u1_tstat, l1dist_u1_pvalue, emdy_u1_tstat, emdy_u1_pvalue,  
     emdx_u1_tstat, emdx_u1_pvalue, l1trans_u1_tstat, l1trans_u1_pvalue]
writer.writerow(s)
s = ["t1u1-t2u1 vs. t1u1-t1u2", l1dist_t1u1_tstat, l1dist_t1u1_pvalue, emdy_t1u1_tstat, emdy_t1u1_pvalue, 
     emdx_t1u1_tstat, emdx_t1u1_pvalue, l1trans_t1u1_tstat, l1trans_t1u1_pvalue]
writer.writerow(s)
f.close()
