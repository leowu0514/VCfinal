
Folder structure:
.
├── (data/)
│   ├── akiyo_qcif.yuv
|   ├── ...
├── src/
│   ├── data_processing.py
│   ├── test.py
│   ├── train.py
│   └── requirements.txt
└── slide/
    ├── VC Project Proposal.pptx
    └── VC Project Presentation.pptx

Something Inportant:
  1. Since we had some problems to embed the videos in `VC Project Presentation.pptx`, so we provide the link of the presentation slide on Google Doc (you can play the videos online while checking the slide) & the link of the videos on Google Drive (you can play each video seperately):
    - the link of our presentation slide: https://docs.google.com/presentation/d/1VdmH5269FHFkKRVTZsuJL0QuK9hNvB_2DrdHZ-pq7TU/edit#slide=id.p
    - the link of our videos: https://drive.google.com/drive/folders/1dn2IibnaGmRplPir65v5cryrNohXhuUJ

  2. Since the `data/` directory which contains training & testing data is too big to upload to Ceiba, so we provide the github repository of our VC final project if you want to check them.
    - the link of our VC final project github repository: https://github.com/leowu0514/VCfinal

Usage:
  Training:
  - pip3 install -r requirements.txt (check if tools are installed)
  - python3 train.py <Y_ratio> <Cb_ratio> <Cr_ratio>
  - <ratio> can be 2, 4, 8 or 16!

  After training, there will be 3 models saved as Y.h5, Cb.h5, and Cr.h5. Then we can start testing!

  Testing:
  - python3 test.py

  After testing, a result will be output (result.qcif), thats the result of our model!