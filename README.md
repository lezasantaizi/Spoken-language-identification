# Spoken-language-identification
方言分类，pytorch
参考的是[topcoder比赛](http://yerevann.github.io/2016/06/26/combining-cnn-and-rnn-for-spoken-language-identification/)的结果

见[知乎](https://zhuanlan.zhihu.com/p/45104360)

## topcoder 原始语音数据
[Training Data Files]( http://www.topcoder.com/contest/problem/SpokenLanguages2/trainingdata.zip)

[Testing Data Files]( http://www.topcoder.com/contest/problem/SpokenLanguages2/testingdata.zip)

[Training Dataset List]( http://www.topcoder.com/contest/problem/SpokenLanguages2/trainingData.csv)

[Testing Dataset List]( http://www.topcoder.com/contest/problem/SpokenLanguages2/testingData.csv)
## 函数介绍：

1. create_spectrograms.py 该函数将语音转成声谱图
* --input_dir:输入的音频文件所在目录
* --save_img_dir:音频转为声谱图之后保存的图片目录
* --audio_type:音频类型（mp3,wav）

2. logger.py 是用于tensorboard可视化的函数

3. main_topcoder.py 是主函数
* batch_gen_imgdata 用于批量读取图片，生成pytorch可以读取的向量
* Network_CNN_RNN 是模型结构类，其中卷积处理后的维度和theano略有不同，不过不影响结果，该结构针对的是topcoder的语音，语音长度固定是10s，
生成的声谱图的维度是 256 * 858，作为pytorch输入的维度是 100 * 1 * 256 * 858


## tips：
1.每个epoch计算dev集合的acc，根据dev的acc结果保存model

2.使用tensorboard可视化训练集的loss和acc，测试集的acc

3.rnn_input_size 的大小需要提前设定好，当声谱图大小有修改或者conv有变动时，该值也需要修改

## 运行
trainEqual.csv格式如下：
>
/movie/audio/topcoder/topcoder_train_png/000w3fewuqj.png,57
/movie/audio/topcoder/topcoder_train_png/000ylhu4sxl.png,55
/movie/audio/topcoder/topcoder_train_png/0014x3zvjrl.png,155
/movie/audio/topcoder/topcoder_train_png/001xjmtk2wx.png,148
/movie/audio/topcoder/topcoder_train_png/002hrjhbsnk.png,110

python -u main_topcoder.py --mode=train --datalist_path=/movie/audio/topcoder --use_gpu=1 --use_pretrained=0
