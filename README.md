# Auxcoformer: Auxiliary and Contrastive Transformer for Robust Crack Detection in Adverse Weather Conditions 
Jae Hyun Yoon, Jong Won Jung, Seok Bong Yoo*

(Abstract) Crack detection is an important part of civil infrastructure care, with the utilization of automated robots for detailed inspections and repairs becoming increasingly common. In the case of autonomous vehicles, ensuring the fast and accurate detection of cracks is crucial for safe road navigation. In these fields, existing models demonstrate impressive performance. However, they are mainly optimized for clear weather and struggle with issues like occlusions and brightness variations in adverse weather conditions. These issues especially affect for automation robots and autonomous vehicle navigation that need to operate reliably in a variety of environmental conditions. To address this problem, we propose Auxcoformer designed for robust crack detection, even in adverse weather conditions. Considering the degradation in images caused by adverse weather conditions, Auxcoformer incorporates an auxiliary restoration network. This network efficiently restores damaged crack details, ensuring the primary detection network obtains better quality features. Additionally, our approach utilizes a non-local patch-based 3D transform technique, which further refines and emphasizes the characteristics of cracks, making them more distinguishable. Considering the connectivity of crack, we also introduce contrastive patch loss for precise localization. Then, we demonstrate the performance of Auxcoformer comparing it with other detection models through experiments.
![Fig2 (1)](https://github.com/jhyoon964/Auxcoformer/assets/144157648/a6d010ef-f3d1-4bf3-96d1-4d6b624c17e4)
