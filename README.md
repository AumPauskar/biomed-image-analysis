# Biomed image analysis

## Before you start
The project is made with python native within **WSL (Ubuntu 22.04.3 LTS)** ver `3.10.12`. All the libraries and dependencies can be found within the subprojects folder and be named `requirements.txt`. You can install it by running the following command:
```bash
pip install -r requirements.txt
```

### MNIST analysis
- Before you run \
	Install all the dependencies from `requirements.txt` \
	There are two variations of the project
	1. **CUDA version:** This version uses GPU to train the model. You need to have a CUDA compatible GPU and CUDA toolkit installed. Most NVIDIA GPUs are CUDA compatible. You can check your GPU compatibility [here](https://developer.nvidia.com/cuda-gpus). You can install CUDA toolkit from [here](https://developer.nvidia.com/cuda-downloads). \
	**Note:** If you have a CUDA compatible GPU, then it's not recommended to run `requirements.txt` as it will install the CPU version of the libraries. To install the GPU version of pytorch [visit here](https://pytorch.org/get-started/locally/).
	2. **CPU version:** This version uses CPU to train the model. You don't need to have a CUDA compatible GPU or CUDA toolkit installed. You can run this version on any machine.
- Testing environment
	|  |  |  |
	| --- | --- | --- |
	| 1 | CPU | Ryzen 7 5800H |
	| 2 | GPU | RTX 3060 Laptop |
	| 3 | RAM | 2x8GB DDR4 @ 3200MHz |
	| 4 | OS | Windows 11/Ubuntu 22.4 WSL |
	| 5 | CUDA | 11.8 |
	| 6 | Python | 3.10.12 |

	**CPU version:** ~ 3.1 mins \
	**CUDA version:** ~ 55 secs

## About the college report
The report source code and all the related files are present in `oba/` and `course_activity` respectively. The report is written in markdown and converted to pdf using pandoc, using the command:
can be conveted into docx using (requires pandoc `sudo apt install pandoc -y`)
```bash
pandoc report.md -o report.docx
```
or can be converted into pdf using (requires texlive `sudo apt install texlive -y`)
```bash
pandoc report.md -o report.pdf
```
