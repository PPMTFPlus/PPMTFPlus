#!/bin/bash -x

cd data/PF/
tar -zxvf traces_TK.tar.gz
cd ../../python/
python3 MakeTrainTestData_PF.py TK
python3 MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
cd ../python/
python3 SampleParamA.py PF TK
cd ../
./SynData_PPMTF PF TK 10 SmplA1 200
cd ../python/
python3 EvalUtilPriv.py PF TK PPMTF 10 SmplA1
#python3 EvalUtilPriv.py PF TK PPMTF 10 A
#python3 EvalUtilPriv.py PF TK SGD 10
#python3 EvalUtilPriv.py PF TK SGLT 10
