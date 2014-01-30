# [Currency Recognition](http://carlosmccosta.github.io/Currency-Recognition/)


## Overview
This project focus on the detection and recognition of Euro banknotes and has the following associated paper:

[Multiview banknote recognition with component and shape analysis](https://github.com/carlosmccosta/Currency-Recognition/raw/master/Report/Multiview%20banknote%20recognition%20with%20component%20and%20shape%20analysis.pdf)

**Abstract**
Robust banknote recognition in different perspective 
views and in dynamic lighting conditions is a critical component in 
assistive  systems  for  visually  impaired  people.  It  also  has  an 
important  role  in  improving  the  security  of  ATM  maintenance 
procedures  and  in  increasing  the  confidence  in  the  results
computed by  automatic banknote counting machines.  Moreover, 
with  the  proper  hardware,  it  can  be  an  effective  way  to  detect 
counterfeit  banknotes.  With  these  applications  in  mind,  it  was 
developed  a  system  that  can  recognize  multiple  banknotes  in 
different perspective views and scales, even when they are part of 
cluttered environments in which the lighting conditions may vary 
considerably. The system is also able to recognize banknotes that 
are partially visible, folded, wrinkled or even worn by usage.  To 
accomplish  this  task,  the  system  is  based  in  image  processing 
algorithms, such as feature detection, description and matching. 
To improve the confidence in the recognition results, the contour 
of the banknotes is computed using a homography, and its shape 
is analyzed to make sure that it belongs to a banknote. The system 
was  tested  with  82  test  images,  and  all  Euro  banknotes  were 
successfully recognized, even when there were several  banknotes 
in the same test image, and they were partially occluded.

## Results

![Fig. 1 - Detection of a banknote in an ideal perspective view](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/5__(5).jpg___SIFT-Detector_SIFT-Extractor_BF-Matcher_lowQualityImageDB_globalMatch__inliersMatches__0.jpg)

Fig. 1 - Detection of a banknote in an ideal perspective view


![Fig. 2 - Detection of a banknote with perspective distortion](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/5__(6).jpg___SURF-Detector_SURF-Extractor_BF-Matcher_lowQualityImageDB_globalMatch__inliersMatches__0.jpg)

Fig. 2 - Detection of a banknote with perspective distortion


![Fig. 3 - Detection of a banknote in cluttered environments](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/10__(9).jpeg___SIFT-Detector_SIFT-Extractor_BF-Matcher_lowQualityImageDB_globalMatch__inliersMatches__0.jpg)

Fig. 3 - Detection of a banknote in cluttered environments


![Fig. 4 - Detection of partially occluded banknotes](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/500.jpg___GFTT-Detector_SIFT-Extractor_BF-Matcher_dynamicQualityImageDB_globalMatch__inliersMatches__0.jpg)
![Fig. 4 - Detection of partially occluded banknotes](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/50__(13).jpg___SIFT-Detector_SIFT-Extractor_BF-Matcher_mediumQualityImageDB_globalMatch__inliersMatches__0.jpg)

Fig. 4 - Detection of partially occluded banknotes


![Fig. 5 - Detection of overlapping banknotes](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/10-20-50.jpg___SIFT-Detector_SIFT-Extractor_BF-Matcher_dynamicQualityImageDB_globalMatch__inliersMatches__1.jpg)
![Fig. 5 - Detection of overlapping banknotes](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/10-20-50.jpg___SIFT-Detector_SIFT-Extractor_BF-Matcher_dynamicQualityImageDB_globalMatch__inliersMatches__2.jpg)
![Fig. 5 - Detection of overlapping banknotes](https://raw.github.com/carlosmccosta/Currency-Recognition/master/Results/Representative%20results/10-20-50.jpg___SIFT-Detector_SIFT-Extractor_BF-Matcher_dynamicQualityImageDB_globalMatch__inliersMatches__0.jpg)

Fig. 5 - Detection of overlapping banknotes


## Releases
[Windows 8.1 release](https://github.com/carlosmccosta/Currency-Recognition/releases)



## Building and developing
The setup instructions on how to build and develop in Visual Studio is available [here](https://github.com/carlosmccosta/Currency-Recognition/blob/master/CurrencyRecognition/docs/Visual%20Studio%20configuration%20for%20OpenCV%202.4.8.txt)
