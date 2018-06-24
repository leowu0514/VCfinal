Folder structure:
.
├── data
│   ├── akiyo_qcif.yuv
|	├── ...
└── src
    ├── data_processing.py
    ├── test.py
    └── train.py

Usage:
	Training:
	-	pip3 install -r requirement.txt (check if tools are installed)
	-	python3 train.py <Y_ratio> <Cb_ratio> <Cr_ratio>
	-	<ratio> can be 2, 4, 8 or 16!

	After training, there will be 3 models saved as Y.h5, Cb.h5, and Cr.h5. Then we can start testing!

	Testing:
	-	python3 test.py

	After testing, a result will be output (result.qcif), thats the result of our model!