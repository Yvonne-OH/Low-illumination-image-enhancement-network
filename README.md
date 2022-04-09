# Low-illumination-image-enhancement-network

This paper proposes a lightweight low-illumination image enhancement network inspired by the Retinex theory. In the model proposed, a pixel-wise adjustment function realizes the lightweight of the network structure; the optimization bottleneck problem is solved by introducing the shortcut mechanism. Through the test, the SSIM index of the proposed model is 7.04% higher than that of MSRCR, and 31.03% higher than that of CLAHE. In the actual use case, the proposed model can process videos with a resolution of 400×600 at a speed of 20fps on average, which meets the requirements of DMS video stream processing speed. Also, a MobileNet distraction state recognition network pre-trained on the SFD dataset is used as the back-end to verify its application in the DMS system. The results show that the recognition accuracy of the driver's distracted behavior in a low-light environment is improved by 75.39% compared to before use.

To run the program, PLZ download from the master branch

![Figure 2022-03-17 170959 (0)](https://user-images.githubusercontent.com/58218024/162571703-59096cc5-6624-4d94-b85a-c3d2de902b8c.png)
![Figure 2022-03-17 170959 (10)](https://user-images.githubusercontent.com/58218024/162571826-853a627f-18a5-4452-89c8-c7cf8e7beaca.png)
![Figure 2022-03-17 170959 (20)](https://user-images.githubusercontent.com/58218024/162571908-a9121983-465e-4f78-9a74-5f0c8ff517de.png)

