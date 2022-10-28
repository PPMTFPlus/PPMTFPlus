#!/bin/bash -x


cd data/PF/
tar -zxvf traces_TK.tar.gz
cd ../../python/
python3.10 MakeTrainTestData_PF.py TK
python3.10 MakeTrainTensor.py PF TK
cd ../cpp/
./PPMTF PF TK 200
cd ../python/
python3.10 SampleParamA.py PF TK
cd ../cpp
./SynData_PPMTF PF TK 10 SmplA1 200
cd ../python/
python3.10 EvalUtilPriv.py PF TK PPMTF 10 SmplA1
