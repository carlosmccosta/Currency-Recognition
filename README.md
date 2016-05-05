# [Currency Recognition](http://carlosmccosta.github.io/Currency-Recognition/)


## Overview
This project focus on the detection and recognition of Euro banknotes and has the following associated resources:

- ICARSC 2016 paper: [Recognition of Banknotes in Multiple Perspectives Using Selective Feature Matching and Shape Analysis](https://www.researchgate.net/publication/301888929_Recognition_of_Banknotes_in_Multiple_Perspectives_Using_Selective_Feature_Matching_and_Shape_Analysis)


- [Presentation](https://www.researchgate.net/publication/301888713_Recognition-of-Banknotes-in-Multiple-Perspectives-Using-Selective-Feature-Matching-and-Shape-Analysis-Presentation)


- [Poster](https://www.researchgate.net/publication/301888902_Recognition-of-Banknotes-in-Multiple-Perspectives-Using-Selective-Feature-Matching-and-Shape-Analysis-Poster)


**Abstract:**
Reliable banknote recognition is critical for detecting counterfeit banknotes in ATMs and help visual impaired people. To solve this problem, it was implemented a computer vision system that can recognize multiple banknotes in different perspective views and scales, even when they are within cluttered environments in which the lighting conditions may vary considerably. The system is also able to recognize banknotes that are partially visible, folded, wrinkled or even worn by usage. To accomplish this task, the system relies on computer vision algorithms, such as image preprocessing, feature detection, description and matching. To improve the confidence of the banknote recognition the feature matching results are used to compute the contour of the banknotes using an homography that later on is validated using shape analysis algorithms. The system successfully recognized all Euro banknotes in 80 test images even when there were several overlapping banknotes in the same test image.

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


## Related repositories
- [ICARSC 2016 latex paper](https://github.com/carlosmccosta/Currency-Recognition-Article)
- [ICARSC 2016 latex presentation](https://github.com/carlosmccosta/Currency-Recognition-Presentation)
- [ICARSC 2016 latex poster](https://github.com/carlosmccosta/Currency-Recognition-Poster)
