# 监仓目标检测



## 数据获取

​		1, HSF文件播放

​				W:\1205ToXiongBiao\YuJinyong\dataplay\decodetool.exe

​		2，HSF文件单帧Depth获取

​				W:\1205ToXiongBiao\YuJinyong\JCDete\DataProcess\GenImageFromHSF.py

​		3，单帧Depth文件转为配准后Colormap文件

​				W:\1205ToXiongBiao\YuJinyong\JCDete\DataProcess\ColormapProcessFuns.py

## 数据标注

​		1，类别: lying, sitting, standing

​		2，vott标注获取的图片

​				xml文件转txt文件： W:\1205ToXiongBiao\YuJinyong\JCDete\DataProcess\LabelProcessFuns.py

## 数据训练

​		1，yolov3/yolov5, pytorch环境搭建

​		2，训练标注的图片数据

​		3，统计验证精度

​		4，选择目标检测模型

## 告警功能逻辑

​		1，获取单传感器各目标位置和类别

​		2，节点端融合各传感器目标

​		3，根据融合后目标的空间位置和状态类别，时序计算告警

​				告警类别：未在指定时间休息、厕所区域异常、单人留仓、超高等

## 在线告警测试

​		1，根据在线测试，查看各告警状态正确性





# RGBD目标检测

## 数据获取

​		[W:\1205ToXiongBiao\YuJinyong\JCDete\DataProcess\DatasetRGBDProcess_EK.py]

​		1，EK告警数据地址：W:\1205ToXiongBiao\YuJinyong\Data\AlarmData\EK

​		2，从告警数据中获取Depth和RGB数据：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\dataset_test

​		3，（如果在以前的模型基础上挑选检测不好的数据）测试dataset_test数据：

见测试数据文档  W:\1205ToXiongBiao\YuJinyong\JCDete\RGBD_Dete\RGBDHumanAnalysis\RGBD检测文档说明.doc	

​				原始数据地址：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\SelectHardCaseImage

​		4，从上述测试的结果中挑选检测不好的数据：

​				挑选数据地址： W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\Colormap

W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\RGB

​		5，标注检测不好的数据

​				标注后的文件地址：

W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\Colormap_Xml

W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\RGB_Xml

​		6，转换标注好的label

​				转换的label地址：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\Colormap_Txt        和        RGB_Txt

​		7，转换RGB数据，并显示两类数据，便于查看标注和转换是否正确

​				转换后数据地址：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\DatasetAddHardImage\CombineRGBDLabel\RGB_MatchToDepth             和     RGBLabel_MatchToDepth

​				显示合并后的数据：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\DatasetAddHardImage\CombineRGBDLabel\ColormapAnnoPlot          和      RGBAnnoPlot       和     RGBAnnoPlot_MatchToDepth

​		8，生成RGBD数据和label

​				生成的RGBD数据地址： W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\DatasetAddHardImage\CombineRGBDLabel\RGBD      和         RGBDLabel

​		9， 生成数据集trainval	

​				生成数据集地址：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\DatasetAddHardImage\Dataset\data_class3

​				（然后，可以将此次标注的图片和之前的图片合并，进行测试和训练）

## 数据标注

​		1，标注数据地址：W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\Colormap

W:\1205ToXiongBiao\YuJinyong\JCDete\Data\Dataset_RGBD\EK_RGBD_202201\AnnoImageVott\RGB

## 数据训练

​		1，测试数据说明： W:\1205ToXiongBiao\YuJinyong\JCDete\RGBD_Dete\RGBDHumanAnalysis\RGBD检测文档说明.doc

## 数据测试

​		1，测试数据说明： W:\1205ToXiongBiao\YuJinyong\JCDete\RGBD_Dete\RGBDHumanAnalysis\RGBD检测文档说明.doc

























