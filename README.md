# PPMTF+
This is a source code of PPMTF+ (Privacy-Preserving Multiple Tensor Factorization Plus) in the following paper:

Takao Murakami, Hiromi Arai, Koki Hamada, Takuma Hatano, Makoto Iguchi, Hiroaki Kikuchi, Atsushi Kuromasa, Hiroshi Nakagawa, Yuichi Nakamura, Kenshiro Nishiyama, Ryo Nojima, Hidenobu Oguri, Chiemi Watanabe, Akira Yamada, Takayasu Yamaguchi, Yuji Yamaoka, "Designing a Location Trace Anonymization Contest," Proceedings on Privacy Enhancing Technologies (PoPETs), Issue 1, 2023 (to appear).

PPMTF+ is a location synthesizer for an anonymization contest and is mostly implemented with C++. Data preprocessing and evaluation are implemented with Python. This software is released under the MIT License.

# Purpose

The purpose of this source code is to reproduce experimental results of PPMTF+ in PF (SNS-based people flow data) and FS (Foursquare dataset). In particular, we designed our code to easily reproduce experimental results of PPMTF+ in PF (Figure 18 "PPMTF+" in our paper) using Docker files. See **Running Our Code Using Dockerfiles** for details.

We also designed our code to reproduce experimental results of PPMTF+ in FS (Figures 16 and 17 "PPMTF+" in our paper) by downloading the Foursquare dataset and running our code. Note that it takes a lot of time (e.g., it may take more than one day depending on the running environment) to run our code. See **Usage (4)(5)** for details.

Note that the purpose of this code is to generate **datasets** for an anonymization contest. To obtain the contest results of our paper (e.g., Figures 8 and 9), please download the "Dataset and Sample program" and "Submitted Files by All Teams (Final Round)" (~1.0GB zipped) from [PWSCup 2019](https://www.iwsec.org/pws/2019/cup19_e.html) and see README files therein.

# Directory Structure
- cpp/			&emsp;C++ codes (put the required files under this directory; see cpp/README.md).
- data/			&emsp;Output data (obtained by running codes).
  - PF/			&emsp;Output data in PF (SNS-based people flow data).
  - PF_dataset/		&emsp;Place PF (SNS-based people flow data) in this directory (currently empty).
  - FS/			&emsp;Output data in FS (Foursquare dataset).
  - FS_dataset/		&emsp;Place FS (Foursquare dataset) in this directory (currently empty).
- python/		&emsp;Python codes.
- results/		&emsp;Experimental results.
  - PF/			&emsp;Experimental results in PF (SNS-based people flow data).
  - FS/			&emsp;Experimental results in FS (Foursquare dataset).
- docker-compose.yml		&emsp;docker-compose.yml file.
- Dockerfile		&emsp;Dockerfile.
- LICENSE.txt		&emsp;MIT license.
- NOTICE.txt		&emsp;Copyright information.
- README.md		&emsp;This file.
- run_PPMTF_PF.sh	&emsp;Shell script to synthesize traces in PF using PPMTF.

Instructions for installing Eigen 3.3.7, Generalized Constant Expression Math, and StatsLib are given in cpp/README.md.

# Running Our Code Using Docker Files

NOTE: We used a laptop with Windows 11 Enterprise, Intel i7-1185G7 (3.00GHz, 4 cores), and 32GB RAM to run our code using Docker files. It required about 200MB memory and about 15 minutes in our environment.

You can easily build and run our code using Docker files as follows.

1. Install [docker](https://docs.docker.com/get-docker/) & [docker-compose](https://docs.docker.com/compose/install/).

2. Clone this repository.
```
$ git clone https://github.com/PPMTFPlus/PPMTFPlus
```

3. Build a docker image and start ("docker compose" may require sudo).
```
$ cd PPMTFPlus
$ docker compose build
$ docker compose up -d 
```

4. Attach to the docker container.
```
$ docker compose exec ppmtfplus bash --login
```

PPMTFPlus repository is cloned in /opt folder. Files can be shared between a host system and the docker container by putting the files in /share folder.

5. Run our code (it took about 15 minutes in our environment).
```
$ cd opt/PPMTFPlus/
$ chmod +x run_PPMTFPlus_PF.sh
$ ./run_PPMTFPlus_PF.sh
```

Then experimental results of PPMTF (alpha=200) in PF will be output in "data/PF/utilpriv_PPMTF_TK_SmplA1.csv" (note that this file is newly created after running our code).

We plotted Figure 18 "PPMTF+" in our paper using this file, while changing the alpha parameter from 0.5 to 200. To see the figure, see "results/PF/utilpriv_PF.xlsx". To change the alpha parameter, see **Usage (3)**.

# Usage

NOTE: (1), (2), and (3) are for reproducing results in PF and can be run using Docker Files. (4) and (5) are for reproducing results in FS and are not covered by Docker Files.

**(1) Install**

Install Eigen 3.3.7, Generalized Constant Expression Math, and StatsLib (see cpp/README.md).

Install our source code (C++) as follows.
```
$ cd cpp/
$ make
$ cd ../
```

Install Python 3.10.
```
$ wget https://www.python.org/ftp/python/3.10.4/Python-3.10.4.tar.xz
$ tar xJf Python-3.10.4.tar.xz
$ cd Python-3.10.4
$ yum install -y epel-release
$ yum install -y openssl11 openssl11-devel
$ export CFLAGS=$(pkg-config --cflags openssl11)
$ export LDFLAGS=$(pkg-config --libs openssl11)
$ ./configure
$ make
$ make altinstall
$ cd ../
```

Install [POT 0.8.2](https://pythonot.github.io/index.html).
```
$ cd python/
$ pip3.10 install POT==0.8.2
$ cd ../
```

**(2) Download and preprocess PF**

NOTE: In (2), we explain how to obtain the POI file (POI_TK.csv) and the trace file (traces_TK.csv) from the [SNS-based people flow data](https://nightley.jp/archives/1954/). The website of the SNS-based people flow data is written in Japanese, and you need to input your information (e.g., affiliation, e-mail address) to download the dataset. However, both the POI and trace files have already been included in data/PF, and (3) uses only these data. Therefore, you can skip (2), and in that case, you do not need to download the dataset.

Download the [SNS-based people flow data](https://nightley.jp/archives/1954/) and place the dataset in data/PF_dataset/.

Run the following commands.

```
$ cd python/
$ python3.10 Read_PF.py ../data/PF_dataset TK
$ cd ../
```

Then the POI file (POI_TK.csv) and the trace file (traces_TK.csv) are output in data/PF/.

**(3) Synthesizing traces for each team in PF**

Run the following commands.

```
$ cd python/
$ python3.10 MakeTrainTestData_PF.py TK
$ python3.10 MakeTrainTensor.py PF TK
$ cd ../cpp/
$ ./PPMTF PF TK 200
(To change the alpha paramter from 200 to [alpha], run "./PPMTF PF TK [alpha]".)
$ cd ../python/
$ python3.10 SampleParamA.py PF TK
$ cd ../
```

Then a generative model for teams 1 to 10 (modelparameter_Itr100_SmplA[1-10].csv) will be generated in data/PF/PPMTF_TK_alp200_mnt100_mnv100/.

To generate a synthetic trace of team 1, run the following commands.

```
$ cd cpp/
$ ./SynData_PPMTF PF TK 10 SmplA1 200
(To change the team number from 1 to [t], run "./SynData_PPMTF PF TK 10 SamplA[t] 200".)
$ cd ../
```

Then synthesize traces (syntraces_Itr100_SmplA1.csv) will be generated in data/PF/PPMTF_TK_alp200_mnt100_mnv100/.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```
$ cd python/
$ python3.10 EvalUtilPriv.py PF TK PPMTF 10 SmplA1
$ cd ../
```

Then experimental results of PPMTF (utilpriv_PPMTF_TK.csv) will be output in data/PF/.

We plotted Figure 18 "PPMTF+" in our paper using this file, while changing the alpha parameter from 0.5 to 200. See "results/PF/utilpriv_PF.xlsx" for details.

**(4) Download and preprocess FS**

Download the [Foursquare dataset (Global-scale Check-in Dataset with User Social Networks)](https://sites.google.com/site/yangdingqi/home/foursquare-dataset) and place the dataset in data/FS_dataset/.

Run the following command to fix garbled text.

```
$ cd data/FS_dataset/
$ cat raw_POIs.txt | sed -e "s/Caf[^.]*\t/Caf\t/" > raw_POIs_fix.txt
$ cd ../
```

Run the following commands.

```
$ cd python/
$ python3.10 Read_FS.py ../data/FS_dataset/ NY
$ cd ../
```

Then the POI file (POI_NY.csv) and the trace file (traces_NY.csv) are output in data/PF/.

The POI file and trace file in other cities (IST/JK/KL/SP/TKY) can also be generated by replacing NY with IS, JK, KL, SP, or TK.

**(5) Synthesizing traces for each team in FS**

Run the following commands.

```
$ cd python/
$ python3.10 MakeTrainTestData_FS.py NY
$ python3.10 MakeTrainTensor.py PF NY
$ cd ../cpp/
$ ./PPMTF FS NY 200
$ cd ../python/
$ python3.10 SampleParamA.py FS NY
$ cd ../
```
Then a generative model for teams 1 to 10 (modelparameter_Itr100_SmplA[1-10].csv) will be generated in data/FS/PPMTF_NY_alp200_mnt100_mnv100/.

To generate a synthetic trace of team 1, run the following commands.

```
$ cd cpp/
$ ./SynData_PPMTF FS NY 20 SmplA1 200
(To change the team number from 1 to [t], run "./SynData_PPMTF PF TK 20 SamplA[t] 200".)
$ cd ../
```

Then synthesize traces (syntraces_Itr100_SmplA1.csv) in NYC will be generated in data/FS/PPMTF_NY_alp200_mnt100_mnv100/.

To evaluate the diversity of the synthetic traces, run the following command.

```
$ python3.10 EvalDiversity.py FS NY SmplA 20 1000
```

Then experimental results of PPMTF+ (div_NY_SmplA_tn20_sn1000.csv) will be output in data/FS/.

To evaluate the utility and privacy of the synthetic traces, run the following command.

```
$ python3.10 EvalUtilPriv.py FS NY PPMTF 20 SmplA1 100 0
```

Then experimental results of PPMTF+ (utilpriv_PPMTF_NY_SmplA1.csv) will be output in data/FS/.

Synthesized traces in other cities (IST/JK/KL/SP/TKY) can also be generated and evaluated by replacing NY with IS, JK, KL, SP, or TK.

We plotted Figures 16 and 17 "PPMTF+" in our paper using these files. See "results/FS/div_FS.xlsx" and "results/FS/utilpriv_FS.xlsx" for details.

**(6) Experimental Results for Other Synthesizers**

To obtain experimental results for other synthesizers, see OtherSynthesizers.md.

# Execution Environment
We used CentOS 7.5 with gcc 11.2.0 and python 3.10.4.

# External Libraries used by PPMTF+
- [Eigen 3.3.7](http://eigen.tuxfamily.org/index.php?title=Main_Page) is distributed under the [MPL2](https://www.mozilla.org/en-US/MPL/2.0/).
- [Generalized Constant Expression Math](https://www.kthohr.com/gcem.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
- [StatsLib](https://www.kthohr.com/statslib.html) is distributed under the [Apache License 2.0](https://github.com/kthohr/stats/blob/master/LICENSE).
