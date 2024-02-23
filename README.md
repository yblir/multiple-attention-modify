# multiple-attention
原multiple-attention代码缺少说明,没法训练,这里在原结构基础上魔改一版
变更内容如下:
1. train_single_gpu.py 是原训练代码(train_raw.py)的修改,只要配对数据集路径,可直接运行
2. train_distributed.py 是分布式训练代码, 在服务器上能跑, 单机会报错.
3. 数据处理方式参考STIL模型(另一个deepfake鉴伪方案),数据输入是视频, 每次从视频随机抽取几帧
    图片进行处理.配置参数文件params.yaml, 数据增强与读取代码位置在datasets/dataset_change.py, 其中
    路径在dataset_change.py中写死了(懒得写入配置文件啦), 记得手动改成自己的.
4. 当前项目, 数据用的是FF++, 每个视频的每张人脸框已经提取好了, 在weights/ffpp_face_rects.pkl中, 也可以使用其他数据, 提取人脸的模型
    是scrfd_opencv_gpu, 在上面自行修改.
5. 这个模型, 只能说, 可以运行, 混合多个数据集, 在服务器上跑了几十个epoch, real/fake的准确率很难提升, loss也难以收敛. 难道是哪里改错了?
    但是仅在FF++数据上表现又还行!

ps: 项目结束了, 在当前模型花费大量精力, 可是最终还是选了其他模型(基于transformer的,效果也不理想).
   deepfake鉴伪领域最大的问题是不能泛化, 尝试很多模型, 都没啥进展, 这个项目让我深深怀疑人生!