#!/usr/bin/env python3
import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal
import sys

################################# Parameters ##################################
#sys.argv = ["SampleParamA.py", "PF", "TK"]
#sys.argv = ["SampleParamA.py", "FS", "IS"]

if len(sys.argv) < 3:
    print("Usage:",sys.argv[0],"[Dataset] [City] ([Alpha (default: 200)])")
    sys.exit(0)

# Dataset
Dataset = sys.argv[1]
# City
City = sys.argv[2]

# Number of traces per user
Alpha = "200"
if len(sys.argv) >= 4:
    Alpha = sys.argv[3]

# Training user index file (input)
TUserIndexFile = "../data/" + Dataset + "/tuserindex_XX.csv"
# Prefix of the model parameter file (input/output)
ModelParameterFile = "../data/" + Dataset + "/PPMTF_XX_alp" + Alpha + "_mnt100_mnv100/modelparameter"
# Number of columns in model parameters (A, B)
K = 16
# Number of iterations in Gibbs sampling
ItrNum = 100
# Number of sampling A
SmplNum = 10
#SmplNum = 1

########################### Read model parameters #############################
# [output1]: mu_A (K-dim vector)
# [output2]: Lam_A (K x K matrix)
def ReadModelParameters():
    # Read hyper-parameter mu_A
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_mu_A.csv"
    f = open(infile, "r")
    mu_A = np.loadtxt(infile, delimiter=",")
    f.close()

    # Read hyper-parameter Lam_A
    infile = ModelParameterFile + "_Itr" + str(ItrNum) + "_Lam_A.csv"
    f = open(infile, "r")
    Lam_A = np.loadtxt(infile, delimiter=",")
    f.close()

    return mu_A, Lam_A

#################################### Main #####################################
# Fix a seed
np.random.seed(1)

# Replace XX with City
TUserIndexFile = TUserIndexFile.replace("XX", City)
ModelParameterFile = ModelParameterFile.replace("XX", City)

# Number of users --> N
N = len(open(TUserIndexFile).readlines()) - 1

# Read model parameters
mu_A, Lam_A = ReadModelParameters()

# Initialize model parameter A
SmplA = np.zeros((SmplNum, N, K))

# Sample model parameter A from hyper-parameters mu_A & Lam_A
for smpl in range(SmplNum):
    for n in range(N):
#        if n % 10000 == 0:
#            print(n)
        SmplA[smpl, n, :] = multivariate_normal(mu_A, inv(Lam_A))
    print("Sampled new feature vectors --> SmplA" + str(smpl+1))
# Output sampled model parameter A
for smpl in range(SmplNum):
    outfile = ModelParameterFile + "_Itr" + str(ItrNum) + "_SmplA" + str(smpl+1) + ".csv"
    np.savetxt(outfile, SmplA[smpl], delimiter=",")
