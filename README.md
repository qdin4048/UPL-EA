# UPL-EA
Entity alignment (EA) aims at identifying equivalent entity pairs across different knowledge graphs (KGs) that refer to the same real-world identity. It has been a compelling but challenging task that requires the integration of heterogeneous information from different KGs to expand the knowledge coverage and enhance inference abilities. To circumvent the shortage of seed alignments provided for training, recent EA models utilize pseudo-labeling strategies to iteratively add unaligned entity pairs predicted with high confidence to the seed alignments for model training. However, the adverse impact of confirmation bias during pseudo-labeling has been largely overlooked, thus hindering entity alignment performance. To systematically combat confirmation bias, we propose a new Unified Pseudo-Labeling framework for Entity Alignment (UPL-EA) that explicitly alleviates pseudo-labeling errors to boost the performance of entity alignment. UPL-EA achieves this goal through two key innovations: (1) Optimal Transport (OT)-based pseudo-labeling uses discrete OT modeling as an effective means to determine entity correspondences and reduce erroneous matches across two KGs. An effective criterion is derived to infer pseudo-labeled alignments that satisfy one-to-one correspondences; (2) Parallel pseudo-label ensembling refines pseudo-labeled alignments by combining predictions over multiple models independently trained in parallel. The ensembled pseudo-labeled alignments are thereafter used to augment seed alignments to reinforce subsequent model training for alignment inference. The effectiveness of UPL-EA in eliminating pseudo-labeling errors is both theoretically supported and experimentally validated. Our extensive results and in-depth analyses demonstrate the superiority of UPL-EA over 15 competitive baselines and its utility as a general pseudo-labeling framework for entity alignment.

<img width="2155" height="1051" alt="404f2c83b2e016145003b601c85309c" src="https://github.com/user-attachments/assets/dfb4dbf8-7cdb-41cd-a110-0f68ba57f954" />

Environment:\
python--3.9.13;\
pytorch--1.13.0+cu117;\
cudatoolkit--11.7.

Source codes instructions:\
1.Download the "data.zip" from https://drive.google.com/file/d/1PkDscyIt2z4n3D2Qm-tB9x2ZslY9PCN8/view?usp=drive_link ;\
2.Extract "data.zip" to replace the original "data" folder in the depository;\
2.Open the file named "training.py" with Spyder then click "Run file" to reproduce results.
