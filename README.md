# 黑出租预警

## 数据源

- [x] 162 channel id
- [x] 单个视频文件
- [x] 单个帧目录

## 处理流程

- [x] 私家车检测
- [x] 跟踪 (ReID 辅助)
- [x] 车牌识别
- [x] 更新 tracker 状态，计算 lost track
- [x] lost track 车牌投票，写入数据库
- [ ] 对比白名单/黑名单，进行忽略/预警
- [ ] 查询数据库，频繁出现的车辆进行预警

## 安装

[INSTALL.md](docs/INSTALL.md)
