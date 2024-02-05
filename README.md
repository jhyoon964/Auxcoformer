# Auxcoformer: Auxiliary and Contrastive Transformer for Robust Crack Detection in Adverse Weather Conditions

Jae Hyun Yoon, Jong Won Jung, Seok Bong Yoo*

(abstract) Crack detection is integral civil infrastructure maintenance, with automation robots for detailed inspections and repairs becoming increasingly common. Ensuring the fast and accurate crack detection for autonomous vehicles is crucial for safe road navigation. In these fields, existing detection models demonstrate impressive performance. However, they are primarily optimized for clear weather and struggle with occlusions and brightness variations in adverse weather conditions. These problems affect automation robots and autonomous vehicle navigation that must operate reliably in diverse environmental conditions. To address this problem, we propose Auxcoformer designed for robust crack detection in adverse weather conditions. Considering the image degradation caused by adverse weather conditions, Auxcoformer incorporates an auxiliary restoration network. This network efficiently restores damaged crack details, ensuring the primary detection network obtains better quality features. The proposed approach uses a non-local patch-based 3D transform technique, emphasizing the characteristics of cracks and making them more distinguishable. Considering the connectivity of cracks, we also introduce contrastive patch loss for precise localization. Then, we demonstrate the performance of Auxcoformer comparing it with other detection models through experiments. Our source code and demo are available at https://github.com/jhyoon964/Auxcoformer.


![Fig_2](https://github.com/jhyoon964/Auxcoformer/assets/144157648/82200c37-a3ba-46b1-95a4-831d9bd67611)




https://github.com/jhyoon964/Auxcoformer/assets/144157648/c313c38d-f7ec-4f51-bb20-6c2a0da7ad10




## Training & Testing
```
# Train
python train.py

# Test
python val.py
