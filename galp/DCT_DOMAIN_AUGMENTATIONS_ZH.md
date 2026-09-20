# 当前支持的 DCT 域数据增强：实现、像素域对应与示例

状态：当前实现说明。核对日期：2026-09-07。范围：当前工作区中的 GALP Direct-DCT、PLS 训练流程及系数掩码实验。

本文以代码实际执行的操作为准。当前包含：块裁剪和 DCT 缩放、随机裁剪缩放、中心裁剪缩放、水平翻转、14 种 RandAugment 操作、Mixup，以及可用于频率扰动的系数选择。反量化和范围归一化也会说明，但它们是数值预处理，不单独算作随机增强。

“DCT 域支持某个同名操作”不意味着它与常见 RGB 像素域实现逐像素相同。例如，当前 `Contrast` 只缩放亮度 DC 系数，不缩放所有亮度频率；`Posterize` 只量化块均值，不逐像素清除颜色低位。

## 1. 数据表示与比较口径

### 1.1 一个 DCT 系数表示什么

以下用 `F[c, h, w, u, v]` 表示一张图的反量化 DCT 系数：

| 维度 | 含义 |
| --- | --- |
| `c` | 分量；Y 为亮度，Cb/Cr 为色度 |
| `h, w` | 图像中 8×8 DCT 块的行、列 |
| `u, v` | 块内垂直、水平频率，均为 `0…7` |
| `F[..., 0, 0]` | DC 系数，表示块的平均值 |
| 其余 63 个位置 | AC 系数，表示块内变化、纹理和边缘 |

训练输出的标准形状为：

```text
Y:    [B, 1, 28, 28, 8, 8]   → 224×224 的亮度采样网格
CbCr: [B, 2, 14, 14, 8, 8]   → 两个 112×112 的色度采样网格
```

在这个输出网格中，移动 1 个 Y 块对应 8 个亮度像素；移动 1 个色度块对应 16 个亮度像素的空间距离。下文 Y/CbCr 的成对几何例子按这一比例说明。

对 JPEG 使用的正交 8×8 DCT、中心化分量 `p−128`，块均值与 DC 的关系为：

```text
D = F[0,0] = 8 × (mean(p) − 128)
mean(p) = 128 + D/8
```

因此，只给 DC 加 `80`，相当于给该块所有像素加 `10`；把整个块的 64 个系数设为 `0`，相当于把该分量设为常数 `128`。Y/Cb/Cr 都置零时对应中性灰色，而不是黑色。

这些对应关系针对当前系数经 IDCT 得到的 YCbCr 信号。原始 JPEG 的量化误差、反量化后的系数限幅、增强中的整数取整，以及最终显示时的像素限幅和色度插值，都可能引入差异。

### 1.2 数值处理的顺序

JPEG 存储的是量化系数 `q`，必须先乘量化表 `Q`，才能使用本文的 DC 数值解释：

```text
q → 按需选择源系数 → F = clamp(q × Q, −1024, 1016)
  → DCT 裁剪／缩放／水平翻转 → 整数系数
  → RandAugment 入口限幅 → 第一个操作 → 第二个操作
  → z = (F + 4) / 1020 → Mixup → 模型
```

RandAugment 的输入为 `int16`，范围 `[-1024, 1016]`，每个操作结束后再次限幅。涉及浮点计算的结果按实现取整；本文的 `round` 指最近整数取整，恰好半整数时取偶数。几何变换输出先落到 `int16` 表示范围；缩放后系数不必仍在 `[-1024,1016]` 内，训练 RandAugment 入口会重新限制到该范围。

公开 `VALIDATION` profile 输出 FP32 的 `z`。Python 训练增强入口会先做 `round(z×1020−4)` 并限幅；原生 PLS 路径用 `TRAINING_PLS` profile 直接取得整数系数，再执行 CUDA 增强和归一化。

特别注意：原始系数 `F=0` 对应归一化值 `z=4/1020≈0.003922`。不能直接把归一化张量乘零来模拟原始 DCT 系数置零。

## 2. 完整操作清单与提供位置

| 操作 | 修改对象 | 最接近的像素域操作 | 当前提供位置 |
| --- | --- | --- | --- |
| 块裁剪 `Crop` | Y/CbCr 空间块范围 | 块边界对齐的裁剪 | Direct-DCT `transforms` |
| DCT 缩放 `Resize` | 块组织和频率系数 | 图像缩放；使用 DCT 频谱重采样 | 原生网格变换，由 profile 指定输出尺寸 |
| 随机裁剪缩放 `RandomResizedCrop_DCT` | 随机裁剪框，再缩放 | RandomResizedCrop | 训练决策层＋原生变换 |
| 中心裁剪缩放 `ResizedCenterCrop_DCT` | 中心区域，再缩放 | 验证阶段 Resize＋CenterCrop | 验证 profile／训练验证决策层 |
| 水平翻转 `HorizontalFlip` | 块列顺序、水平奇频符号 | HorizontalFlip | Direct-DCT `transforms` |
| `AutoContrast` | Y 的 DC | 按块均值做自动对比度拉伸 | PLS RandAugment |
| `Posterize` | Y/CbCr 的 DC | 块均值量化 | PLS RandAugment |
| `SolarizeAdd` | 负值 Y DC | 给较暗块增加亮度 | PLS RandAugment |
| `Color` | CbCr 的 DC | 块平均色度强度调整 | PLS RandAugment |
| `Contrast` | Y 的 DC | 块均值相对 128 的对比度调整 | PLS RandAugment |
| `Brightness` | Y 的 DC | 整张亮度图加常数 | PLS RandAugment |
| `MidfreqAug` | Y 的全部频率，包括 DC | 块内频率滤波／细节增强 | PLS RandAugment |
| `Cutout` | 矩形区域内的全部系数 | 中性灰色矩形遮挡 | PLS RandAugment |
| `TranslateX` | Y/CbCr 块列 | 整块水平平移＋灰色填充 | PLS RandAugment |
| `TranslateY` | Y/CbCr 块行 | 整块垂直平移＋灰色填充 | PLS RandAugment |
| `Rotate90` | 块位置、频率轴和符号 | 顺／逆时针旋转 90° | PLS RandAugment |
| `AutoSaturation` | CbCr 的 DC | 自动拉伸块平均色度 | PLS RandAugment |
| `Grayscale` | CbCr 的全部系数 | 去色，保留亮度 | PLS RandAugment |
| `ChromaDrop` | Cb 或 Cr 的全部系数 | 移除一个色差信号 | PLS RandAugment |
| `Mixup` | 两张图的全部系数与标签 | 图像和标签的线性混合 | PLS 训练流程 |
| 系数选择／频率掩码 | 每个源块的指定频率 | 分块频率滤波 | 公开 `coefficients` 参数及独立掩码实验 |

这里前四项存在组合关系：随机裁剪缩放和中心裁剪缩放都由裁剪、缩放组成。14 种 RandAugment 操作以训练配方中的列表为准。

公开 `DirectDctReader.read(..., transforms=...)` 的描述符实际解析 `crop`、`horizontal_flip` 以及两个样本追踪字段 `logical_sample_id`、`augmentation_key`。它没有任意颜色增强或 RandAugment 的配置接口；传入 `brightness`、`rotate` 或 `resize` 字段不会新增对应操作。输出尺寸由 profile 决定。

PLS 的 14 种操作既有 Python 标量参考和分组批处理实现，也有原生 CUDA 实现。Python 原生 PLS 适配器位于 `galp.torch.experimental`。仅选择 `TRAINING_PLS` profile 并普通读取，不会自动执行完整 RandAugment 和 Mixup；这些由 PLS pipeline 的后处理负责。

## 3. 几何增强

### 3.1 块裁剪 Crop

**怎么做。** 对 DCT 块网格取子区域，保留被选块内部的 64 个系数。对于 4:2:0 源图，若亮度裁剪框为 `(x,y,width,height)=(32,48,224,224)`，则：

```python
# 单张图，均为反量化、未归一化的 DCT 网格。
y_crop = y[:, 6:34, 4:32, :, :]
cbcr_crop = cbcr[:, 3:17, 2:16, :, :]
```

**像素域对应。** 对应像素图 `image[48:272, 32:256]`。当裁剪框在各分量的块边界对齐、且不需要重采样时，块选择本身与裁剪其 IDCT 图像等价。

**边界。** 当前训练裁剪按 16 个亮度像素对齐，保证 Y 与半分辨率色度对齐。底层会把像素裁剪范围映射成分量块范围；任意非对齐框不应被解释成精确的逐像素裁剪。上例要求源图至少覆盖裁剪范围。

### 3.2 DCT 缩放 Resize

**怎么做。** 使用 DCT 基变换矩阵在相邻小块与大块之间转换，再截取或补齐频率，最后组织回 8×8 块。

- 下采样：将相邻 `L×M` 个 8×8 块组合到一个 `(8L)×(8M)` 的 DCT 表示，保留低频 8×8 部分，并除以 `sqrt(LM)`。
- 上采样：将 8×8 频谱放入更大的频谱，其余位置补零，乘以 `sqrt(LM)`，再分解成 `L×M` 个 8×8 块。
- 一般尺寸比：按最大公约数求每个轴的上／下采样因子；原生实现将这些线性转换合成权重执行。

二维整体关系可写成 `F_out = A_h F_in A_wᵀ`，其中矩阵同时包含块合并／分解、频率选择和尺度补偿。它会混合不同块和不同频率，不是把每个频率当作一张小图做双线性插值。

**像素域对应。** 都在改变图像尺寸，但当前操作对应 DCT 频谱截断／补零定义的重采样，不能宣称与 PIL/torchvision 的 bilinear 或 bicubic resize 完全一致。训练描述符中的 `interpolation="bilinear"` 不会把原生 DCT 执行器改为双线性插值。

**例子。** 裁出的 448×448 亮度区域是 `56×56` 个块，缩为 `28×28` 块后对应 224×224；色度由 `28×28` 块缩为 `14×14`。每个亮度输出块结合 2×2 个源块。若输入区域恒为亮度 `148`，每个源块只有 `DC=160`，缩放后的常值块仍为 `DC≈160`，不会因块数减少而整体变暗。

### 3.3 随机裁剪缩放 RandomResizedCrop_DCT

**怎么做。** PLS 的 published 配方按以下规则选择区域，再使用上一节的 DCT 缩放：

1. 在源 Y 块网格上采样目标面积比例，范围 `[0.05,1.0]`，宽高比固定为 `1`。
2. 将候选边长对齐到适合 DCT 缩放的尺寸。小尺寸候选为 `{2,4,14,28}` 个 Y 块；较大尺寸采用 `28` 的整数倍，并检查源图大小。
3. 随机选起点，再向下对齐到 2 个 Y 块，即 16 个亮度像素。最多尝试 10 次，失败后采用中心区域。
4. 输出固定为 Y `28×28` 块、CbCr `14×14` 块。

**像素域对应。** 对应随机选取图像的一部分并缩放为 224×224。区别是当前 DCT 配方要求块对齐、边长离散化，并固定方形随机候选；不能把 RGB RandomResizedCrop 的连续尺寸和宽高比采样直接视为相同分布。

**例子。** 在 512×512 源图上，`(x,y,width,height)=(16,16,448,448)` 是合法框：选取 Y 的 `[2:58,2:58]` 块，再缩为 `28×28` 块。源图更大时可以出现 `84、112…` 块的候选，不是永远只有四种裁剪大小。

**随机作用范围。** `per-sample` 按种子、epoch、样本身份派生裁剪；`per-pls` 按种子、epoch、PLS 身份派生共享裁剪随机种子。在同尺寸图像上后者得到相同框，异尺寸图像仍需按各自尺寸映射。水平翻转保持按样本派生。基础训练模块还保留一个先在像素坐标采样、再做 16 像素对齐的 v1 配方，以及按 shard 共享裁剪的变体；它们复用同类变换，但随机分布不等于 PLS published 配方。

### 3.4 中心裁剪缩放 ResizedCenterCrop_DCT(32,28)

**怎么做。** 按 `28/32` 计算源块网格的目标裁剪宽高，再选择合适的块数和对齐后的中心位置，最后缩放到 `28×28` 个 Y 块。`32` 是计算中心区域的参考块数，不代表必须先实际生成一张 256×256 像素图。

**像素域对应。** 对应验证阶段“缩放到参考大小，再取中心区域”的用途；块尺寸离散化和 DCT 重采样会影响结果。

**例子。** 512×512 源图有 `64×64` 个 Y 块，`64×28/32=56`。因此取中心 `56×56` 块，即像素框 `(32,32,448,448)`，再缩为 224×224。不是直接从原图中心截取 224×224。

### 3.5 水平翻转 HorizontalFlip

**怎么做。** 反转块列顺序，同时把每个块中水平频率 `v` 为奇数的系数取反：

```text
F'[c,h,w,u,v] = (−1)^v × F[c,h,W−1−w,u,v]
```

**像素域对应。** 对应 `image[:, ::-1]`。前一步交换块的位置，后一步完成每个块内部的左右翻转。只反转块列而不改符号，会留下块内朝向错误。

**例子。** 一行的三个块 `[A,B,C]` 变成 `[flip(C),flip(B),flip(A)]`。原块第一频率行 `[160,20,−10,6,0,0,0,0]` 变为 `[160,−20,−10,−6,0,0,0,0]`。训练概率为 `0.5`，Y/CbCr 使用同一个翻转决定。

在完整块网格上，这一置换和符号规则与像素翻转精确对应；训练操作后的限幅仍可能改变超出范围的系数。

## 4. 14 种 RandAugment 操作

当前固定配方为每张图顺序选择 **2 个操作**，强度为 **11 档中的索引 3**。下面给出的强度来自当前代码，不代表已有公开接口允许任意改档位。`D` 表示被操作分量的 DC，未提及的系数保持原值，最后统一取整／限幅。

### 4.1 AutoContrast：自动拉伸亮度块均值

**怎么做。** 取一张图所有 Y 块 DC 的最小、最大值 `Dmin、Dmax`，仅对 DC 做：

```text
D' = −1024 + (D−Dmin)/(Dmax−Dmin) × 2040
```

**像素域对应。** 最暗块的均值变为 0，最亮块的均值变为 255，其余块均值线性映射；每个块内部的纹理振幅保留。普通像素 AutoContrast 通常按像素极值调整整个通道，二者不等价。

**例子。** 三个块的 `D=[−400,0,400]`，均值 `[78,128,178]`，变换后 `D'=[−1024,−4,1016]`，均值 `[0,127.5,255]`。AC 如 `F[0,1]=20` 保持 20。

**常量输入边界。** 原生 CUDA 在 `Dmin==Dmax` 时保持原值；Python 参考／批处理只专门保护“所有 DC 都为 0”。对于所有 DC 相等且非零的输入，Python 公式会除零，不能宣称与原生路径在该边界等价。`AutoSaturation` 也有相同边界。

### 4.2 Posterize：量化块均值

**怎么做。** 对 Y、Cb、Cr 的 DC 使用 `bitoffset=2`，量化间隔为 `2²=4`：

```text
D' = −1024 + 4 × round((D+1024)/4)
```

**像素域对应。** 把分量块均值量化到间隔 `4/8=0.5` 的等级；AC 不变。不是对 RGB 每个像素做位运算，也不是每个像素“减少 2 bit”。

**例子。** `D=101` 变成 `100`，块均值 `140.625→140.5`；同一块的 AC `37` 仍为 37。

### 4.3 SolarizeAdd：给暗块增加亮度

**怎么做。** 仅检查 Y DC。固定强度 `264.9` 先转整数 `264`，阈值为 `0`：

```text
D' = D+264，若 D<0；否则 D'=D
```

**像素域对应。** 对平均亮度低于 128 的块，整体增加 `264/8=33`；块内每个像素都增加，即使其中某些像素本来较亮。普通逐像素 SolarizeAdd 按各像素阈值选择，结果不同。这里没有执行 Solarize 的像素反相。

**例子。** `D=−224`，均值 100，变成 `D'=40`，均值 133；`D=160`，均值 148，则不变。

### 4.4 Color：调整平均色度强度

**怎么做。** 仅缩放 Cb 和 Cr 的 DC，因子 `a=1±0.27`，即 `0.73` 或 `1.27`：`D'=round(aD)`。

**像素域对应。** 每个色度块均值相对中性值 128 的偏移被缩放，块内色度纹理保持原振幅。对色度分量 `C`，可写成 `C'(x)=C(x)+(a−1)(mean_block(C)−128)`。因此，它类似块平均饱和度调整，并非对所有色度信号统一乘因子。

**例子。** 某块 `(DC_Cb,DC_Cr)=(200,−100)`，色度均值 `(153,115.5)`；取 `a=1.27` 后为 `(254,−127)`，均值约 `(159.75,112.125)`。Y 和色度 AC 都不变。

### 4.5 Contrast：调整亮度块均值对比度

**怎么做。** 仅缩放 Y DC：`D'=round(aD)`，`a∈{0.73,1.27}`。

**像素域对应。** 每个亮度块执行 `Y'(x)=Y(x)+(a−1)(mean_block(Y)−128)`。相当于让块均值远离或接近 128，保留块内偏差。普通逐像素对比度调整会同时缩放 AC 对应的细节，当前实现不会。

**例子。** `D=[−400,0,400]`，取 `a=1.27`，得到 `[−508,0,508]`；块均值 `[78,128,178]→[64.5,128,191.5]`。每个块原有的明暗纹理幅度不变。

### 4.6 Brightness：全图亮度平移

**怎么做。** 先计算一张图所有 Y DC 的平均绝对值 `A=mean(|D|)`，再对所有 Y 块加同一个偏移：`D'=round(D+A×m)`，`m=±0.27`。

**像素域对应。** 给整张亮度图增加常数 `Δ=A×m/8`。这是像素域的加法亮度调整，强度由图像内容决定；不是把 RGB 像素乘以 `1.27`。CbCr 和 Y AC 不变。

**例子。** `D=[−400,0,400]`，`A=800/3`。取 `m=0.27` 得偏移 `72`，输出 `D'=[−328,72,472]`，三个块的每个像素都加 9。若所有 DC 均为 0，则该增强无效果。

### 4.7 MidfreqAug：按频率增强或抑制亮度

**怎么做。** 将每个 Y 块的两条频率轴循环移动 4 个位置，乘二维高斯窗或其倒数，再移回。对 8×8 块和 `m=±0.27`：

```text
σ = 4−2.2×|m| = 3.406
i = (u+4) mod 8，j = (v+4) mod 8
G(u,v) = exp(−((i−3.5)²+(j−3.5)²)/(2σ²))
F'[u,v] = F[u,v]×G(u,v)     当 m<0
F'[u,v] = F[u,v]/G(u,v)     当 m≥0
```

**像素域对应。** 是对每个块的不同 DCT 基函数赋予不同增益，可用于削弱或增强某些纹理。它是块内频率滤波，不等于全图 GaussianBlur 或常规锐化；频率索引循环移位也意味着不能简单称作“频率越高，衰减越强”。

**例子。** 取 `m=−0.27`，`G(0,0)≈0.9787`，`G(4,4)≈0.3479`；若两个位置的系数均为 100，取整后分别约为 `98` 和 `35`。取 `m=+0.27` 时对应约为 `102` 和 `287`。它也修改 DC，并非严格只修改中频。

### 4.8 Cutout：矩形遮挡

**怎么做。** 在 Y 块网格上抽取位置，并将行列位置向下对齐到偶数。强度 `1.8` 经取整和偶数对齐得到 `pad=2`；色度使用一半的位置和 `pad=1`。被选区域内所有频率系数置零。

当前代码保留了参考实现的纵向索引规则。对高宽 `H,W`、抽样位置 `c_h,c_w`、半径 `p`，遮挡范围是：

```text
行：[max(0,H−c_h−p), H−max(0,c_h−p))
列：[max(0,c_w−p),   W−max(0,W−c_w−p))
```

即纵向位置按 `H−c_h` 附近构造，不能直接按 `[c_h−p,c_h+p)` 解释。

**像素域对应。** 在相应位置遮挡一个中性灰色矩形，边缘处截断。不是以黑色 RGB 填充，也不混入其他图片。

**例子。** Y 网格 `28×28`，抽样位置 `(8,10)`，`pad=2`，实际清零块行 `[18,22)`、块列 `[8,12)`，对应亮度像素行 `[144,176)`、列 `[64,96)`，即 32×32 的灰色矩形。CbCr 清零块行 `[9,11)`、列 `[4,6)`，在亮度坐标中覆盖同一区域。

### 4.9 TranslateX：水平平移

**怎么做。** 沿块列方向移动整个块；移入边界的块全部系数填零。Y 位移按下式向下取偶数，CbCr 位移为其一半：

```text
b = 2 × floor(m/2)，m=±3.75
```

**像素域对应。** 整块水平平移并用中性灰色补边。完整块网格上的置换与相同像素距离的平移对应，无亚像素插值。

**例子。** `m=+3.75` 得 `b=+2`，Y 向右 2 块，即 16 像素，CbCr 向右 1 块。`m=−3.75` 得 `b=−4`，Y 向左 32 像素，CbCr 向左 2 块。正负位移不对称，不能写成“左右各 16 像素”或“±30 像素”。

### 4.10 TranslateY：垂直平移

**怎么做。** 与 TranslateX 使用相同强度和取整，只把操作轴改为块行。

**像素域对应。** 整块垂直平移并以中性灰色补边。

**例子。** `m=+3.75` 时 Y 向下 2 块，即 16 像素，顶部填灰；`m=−3.75` 时向上 4 块，即 32 像素，底部填灰。CbCr 相应移动 `+1` 或 `−2` 块。

### 4.11 Rotate90：旋转 90°

**怎么做。** 同时旋转块网格、转置每个块的频率矩阵，并按方向修改奇数频率的符号。对方形网格：

```text
逆时针（m=+1）：F'[h,w,u,v] = (−1)^u F[w,W−1−h,v,u]
顺时针（m=−1）：F'[h,w,u,v] = (−1)^v F[H−1−w,h,v,u]
```

**像素域对应。** 对应图像整体顺／逆时针旋转 90°，无需任意角度插值；系数置换和符号操作在完整块网格上具有精确对应关系。

**例子。** 块网格 `[[A,B],[C,D]]` 逆时针后的位置为 `[[B,D],[A,C]]`，每个块自身也旋转。某源块的 `F[0,1]=20` 在对应旋转块中变成 `F'[1,0]=−20`。当前配方选择的是正负 90°，不是任意角度旋转。

### 4.12 AutoSaturation：自动拉伸平均色度

**怎么做。** 将 `AutoContrast` 的 DC 拉伸公式应用于 CbCr。最小、最大值在同一张图的 Cb 和 Cr 两个通道、所有块之间共同统计；Y 和色度 AC 不变。

**像素域对应。** 自动扩展块平均色差的范围。它没有直接计算 HSV 饱和度，并且可能同时改变色彩平衡，不能解释成标准 HSV Saturation 拉伸。

**例子。** 若所有色度 DC 的全局极值为 `−200` 和 `200`，则 `−200→−1024`、`0→−4`、`200→1016`，对应色度块均值 `103→0`、`128→127.5`、`153→255`。共同使用两通道极值与分别拉伸 Cb、Cr 的结果可能不同。

### 4.13 Grayscale：灰度化

**怎么做。** 将 Cb、Cr 的所有 DC 和 AC 都设为 0，保持 Y 不变。

**像素域对应。** 两个色度分量成为中性常值 128，还原 RGB 后只剩亮度，得到灰度图。不是只去掉色度均值；如果保留色度 AC，仍然会残留彩色纹理。

**例子。** 某块 Y 的均值为 150，不论原 Cb、Cr 如何变化，灰度化后 RGB 三个通道的块均值均约为 150；每个像素满足 `R≈G≈B≈Y`，Y 的纹理仍保留。进入模型前，置零色度系数再归一化为 `4/1020`。

### 4.14 ChromaDrop：丢弃一个色度通道

**怎么做。** 按样本派生的随机决定选择 Cb 或 Cr，将该通道的全部 DC/AC 置零；另一个色度通道和 Y 保留。

**像素域对应。** 去掉蓝色差或红色差信号。它不是 RGB 的 B/R 通道置零，也不等于完整灰度化。

**例子。** 常值区域为 `(Y,Cb,Cr)=(150,158,108)`，即色度 DC 为 `(240,−160)`。丢弃 Cb 后变成 `(150,128,108)`；丢弃 Cr 后变成 `(150,158,128)`。剩余色差信号仍会使图像带颜色。

## 5. Mixup：混合两张图和标签

**怎么做。** 对一个 microbatch 抽取一组混合权重。配方 `alpha=0.2`，两项权重按 `[alpha,alpha]` 的 Dirichlet 分布采样，相当于二元 Beta 混合，再把较大的权重分配给原图。配对方式为 batch 维度 `roll(1)`：

```text
[A,B,C,D] 的配对图为 [D,A,B,C]
λ ≥ 0.5
Z'_i = λ Z_i + (1−λ) Z_partner
t'_i = λ one_hot(label_i) + (1−λ) one_hot(label_partner)
```

所有 Y/CbCr 系数使用同一组权重；同一 microbatch 内共享 λ。原生 PLS 按 microbatch 边界配对，不跨 microbatch；只有一张图时与自己混合。

**像素域对应。** 对齐尺寸和位置后的两张图做透明叠加，并以相同权重混合标签。DCT/IDCT 是线性变换；本项目的归一化是仿射变换，且权重和为 1，因此在无额外取整、限幅以及采用相同色度重建方式时，系数混合与相应 YCbCr 像素混合具有相同线性关系。比较对象是增强后的两张图，不是未经 JPEG 量化的原始 RGB 图。

**例子。** 某对应块在 A 中 `DC=160`，均值 148，在 B 中 `DC=−80`，均值 118。取 `λ=0.75`，混合后 DC 为 `100`，均值 `140.5=0.75×148+0.25×118`。若 A 是猫、B 是狗，标签为 `0.75×猫+0.25×狗`。实际代码在归一化 FP32 系数上混合，无需把混合值重新取整到 int16。

## 6. 系数选择与频率掩码

**怎么做。** 对每个源块使用二值频率掩码 `M[u,v]`，只保留选定系数：`q'=M⊙q`。在当前 RGB-no-more transformed-grid profile 下，未选源系数按零处理，然后再做反量化、裁剪、缩放等频率混合。模型输出仍是每块 8×8 的稠密网格，不会因为选择 16 个源系数就变成 16 维输出。

**像素域对应。** 相当于在每个 JPEG 块内去掉某些空间变化模式。低频保留产生平滑和块状效果；高频或中频保留突出相应纹理，但若去掉 DC，中心化后的块均值为 0，加回 128 后均值为 128。不能把“只保留高频”直接解释成普通 RGB 边缘图。

| 已实现选择方式 | 具体例子 | 像素域效果和使用边界 |
| --- | --- | --- |
| DC-only／低频前缀 | `prefix_k01` 仅保留 DC；`prefix_k16` 保留前 16 个 zigzag 频率 | K=1 在源块上还原为块均值组成的马赛克；K 增大逐步保留更多纹理 |
| 高频窗口 | `high_k04` 保留 zigzag 序号 `60,61,62,63` | 保留最高端频率，移除低频结构和 DC |
| 中频窗口 | `mid_k04` 保留序号 `30,31,32,33` | 只保留中间频带的块内变化 |
| 固定随机子集 | `random_k04` 默认保留 zigzag 序号 `31,41,53,55` | 默认种子为 `20260816`；各 K 使用独立随机流，同一条件对所有图、所有块采用同一掩码 |
| 任意合法子集 | 源列集合 `[0,5,18,63]` | 精确指定保留频率；需先明确源数据的列顺序 |

公开 API 示例：

```python
from galp.torch import DirectDctReader
from galp.profiles.rgbnomore import VALIDATION

reader = DirectDctReader("/path/to/dct/manifest.bin")
dc_only = reader.read([0], VALIDATION, coefficients=[0])
selected = reader.read([0], VALIDATION, coefficients=range(16))
```

`coefficients` 接受 1 至 64 个不重复的 `0…63` 源列索引，`None` 表示全部。它选择的是**源存储系数列**：当前默认 zigzag 存储下 `range(16)` 是 zigzag 前 16 项；若数据使用自然顺序存储，就不是同一频率集合。输出张量最后两维是自然二维频率 `(u,v)`。不能混淆源列序号、zigzag 序号与 `u×8+v`。

例如 zigzag 前六项对应自然索引 `[0,1,8,16,9,2]`，即频率 `(0,0),(0,1),(1,0),(2,0),(1,1),(0,2)`。

独立系数掩码实验已定义 64 个低频前缀条件，以及高频、中频、随机各 4 个条件，共 76 个条件。它们是已实现的选择／评估方式，**没有自动加入默认训练 RandAugment**。固定随机子集也不表示每张图都会重新随机抽取频率。掩码在缩放前应用，缩放会重新混合频率，因此最终输出的非 DC 位置可以重新出现非零值，即使输入只保留了 DC。

## 7. 归一化及默认组合方式

### 7.1 ToRange 是系数预处理

当前归一化为 `z=(F+4)/1020`，逆变换为 `F=1020z−4`：

| 系数 F | 归一化值 z |
| --- | --- |
| `−1024` | `−1` |
| `−4` | `0` |
| `0` | `4/1020≈0.003922` |
| `1016` | `1` |

它在系数数值区间上起到与像素模型输入归一化类似的作用，但不是对图像像素做 `p/127.5−1`。尤其不能把所有 AC 上的 `+4` 解释为图像亮度偏移；亮度平移只对应 DC 的改变。

### 7.2 当前训练／验证流程

| 流程 | 操作顺序 |
| --- | --- |
| 基础训练 v1 | 随机裁剪缩放 → 随机水平翻转 → 范围归一化；该配方未启用 RandAugment 和 Mixup |
| PLS published 训练 | 随机裁剪缩放 → 随机水平翻转 → 2 个 RandAugment 操作 → 范围归一化 → Mixup |
| 验证 | 中心裁剪缩放 `ResizedCenterCrop_DCT(32,28)` → 范围归一化 |
| 系数掩码评估 | 源系数掩码 → 反量化 → 中心裁剪缩放 → 范围归一化 |

PLS RandAugment 按样本身份、种子、epoch、操作序号派生选择和符号，每张图依次做两个操作。第二步处理第一步的结果，并重新计算所需统计量；同一个操作可以重复出现。

配方对颜色操作有互斥规则：选中 `Grayscale` 后，后续候选删除 `Grayscale、Color、AutoSaturation、ChromaDrop`；先选中另外三个色度操作之一后，后续删除 `Grayscale`。按操作分组的批处理保持各样本的决定和执行顺序。

**组合例子。** 对 512×512 图选框 `(16,16,448,448)`，缩到 224×224，再水平翻转；假设两项增强依次选到 `Brightness(+0.27)` 和 `TranslateX(+3.75)`，则先按全图 DC 统计量增亮，再向右平移 16 像素并在左侧填灰。最后归一化，与 microbatch 中前一张图做 Mixup。交换增强顺序可能改变像素结果及后续统计量，因此文档和复现代码应保留顺序。

### 7.3 公开接口：指定裁剪与翻转

以下示例假定样本 0、1 均为 512×512，且已安装匹配的 GALP Torch/CUDA 扩展并准备好 manifest：

```python
from galp.torch import DirectDctReader
from galp.profiles.rgbnomore import VALIDATION

reader = DirectDctReader("/path/to/dct/manifest.bin")
batch = reader.read(
    [0, 1],
    VALIDATION,
    transforms=[
        {
            "crop": {"x": 16, "y": 16, "width": 448, "height": 448},
            "horizontal_flip": True,
            "logical_sample_id": "sample-0",
        },
        {
            "crop": {"x": 32, "y": 32, "width": 448, "height": 448},
            "horizontal_flip": False,
            "logical_sample_id": "sample-1",
        },
    ],
)
y, cbcr = batch.tensors
assert tuple(y.shape) == (2, 1, 28, 28, 8, 8)
assert tuple(cbcr.shape) == (2, 2, 14, 14, 8, 8)
# VALIDATION 给出归一化 FP32 系数。此处还没有 RandAugment 或 Mixup。
```

### 7.4 Python 训练实现：追加 RandAugment 和 Mixup

接续上一例，可通过仓库内的训练辅助函数复现后处理。这些函数属于 benchmark 训练代码，不是 `galp.torch` 的稳定增强接口；其中部分参考操作依赖本地 RGB-no-more 的 `utils.dct_ops`。

```python
from pathlib import Path
import torch
from galp.benchmarks.system_dct_major.training_pls.published_augmentation import (
    apply_published_randaugment,
    apply_published_mixup,
)

augmented, records = apply_published_randaugment(
    (y, cbcr),
    training_seed=11997733,
    epoch=7,
    logical_sample_ids=["sample-0", "sample-1"],
    rgbnomore_root=Path("/home/tangyuxin/RGB-no-more"),  # 改为实际依赖目录。
)
labels = torch.tensor([3, 8], dtype=torch.long, device=y.device)
mixed, soft_targets, mixup_record = apply_published_mixup(
    augmented,
    labels,
    training_seed=11997733,
    epoch=7,
    microbatch_index=0,
    alpha=0.2,
    classes=1000,
)
print(records[0])       # 样本 0 实际选中的两个操作、强度和随机键。
print(mixup_record)     # 原图和配对图的权重。
assert tuple(soft_targets.shape) == (2, 1000)
```

若使用 `galp.torch.experimental.DirectDctPlsPipeline`，其返回的 microbatch 已经过原生 CUDA RandAugment、归一化和 Mixup，并提供 `targets`；无需再调用上述两个后处理函数。完整训练入口见 [PLS 训练说明](benchmarks/system_dct_major/training_pls/README.md)。

## 8. 支持边界与源码索引

当前这套公开 Direct-DCT／PLS 配方没有暴露独立垂直翻转、任意角度旋转、Shear、Perspective、Hue、Gamma、逐像素 Equalize、逐像素 Solarize、GaussianBlur、CutMix 等通用增强接口。外部 RGB-no-more 工具文件中存在其他函数，也不能据此算作已接入 GALP 当前配方。`Rotate90` 内部使用的垂直奇频变号只是旋转的一部分。

最关键的实现区别是：颜色类操作中的 `AutoContrast、Posterize、SolarizeAdd、Color、Contrast、Brightness、AutoSaturation` 只改 DC；`MidfreqAug` 会改 Y 的 DC 和 AC；`Grayscale、ChromaDrop、Cutout` 清除的是对应区域／通道的全部频率。所有“等价”解释都需同时考虑块对齐、色度采样、系数取整和限幅。

| 需要核对的内容 | 代码入口 |
| --- | --- |
| 14 个 RandAugment 名称、固定强度档位、Mixup alpha、训练组合 | [recipe.py](benchmarks/system_dct_major/training_pls/recipe.py) |
| published 裁剪、验证几何、逐项 Python 运算、随机作用范围 | [published_augmentation.py](benchmarks/system_dct_major/training_pls/published_augmentation.py)：`_published_dct_crop_blocks`、`_magnitude`、`_apply_operation_batch`、`apply_published_mixup` |
| 原生 CUDA 的逐项增强、统计、限幅和标签混合 | [direct_dct_pls_postprocess.cu](src/api/direct_dct_pls_postprocess.cu)：`compute_stats_kernel`、`apply_randaugment_kernel`、`normalize_mixup_kernel` |
| 原生随机决定和裁剪生成 | [direct_dct_pls.cpp](src/api/direct_dct_pls.cpp)：`published_randaugment_decision`、`published_crop` |
| profile 几何、数据类型、反量化、源系数缺失语义 | [rgbnomore.hpp](include/galp/profiles/rgbnomore.hpp) |
| 裁剪和 DCT 缩放权重、分量采样比例 | [jpeg_dct_planner.cpp](src/jpeg/jpeg_dct_planner.cpp) |
| GPU 反量化、频率混合、水平翻转、输出取整 | [jpeg_dct_transform_kernels.cu](src/jpeg/jpeg_dct_transform_kernels.cu) |
| 公开 `read`／`pipeline` 参数 | [direct_dct.py](torch/direct_dct.py) |
| `transforms` 实际解析字段 | [direct_dct_torch.cpp](torch/direct_dct_torch.cpp)：`parse_transform_requests` |
| 原生 PLS 的实验性 Python 入口 | [experimental.py](torch/experimental.py) |
| 基础 v1 配方与水平翻转辅助函数 | [augmentation.py](benchmarks/system_rgbnomore/training/augmentation.py) |
| 频率掩码条件定义 | [masks.py](experiments/coefficient_mask_evaluator/masks.py) 与 [实验说明](experiments/coefficient_mask_evaluator/README.md) |
| 已有裁剪、RandAugment 分组／标量一致性测试 | [test_training_pls.py](benchmarks/system_dct_major/tests/test_training_pls.py) |
| 已有原生随机决定与参考结果对照测试 | [direct_dct_pls_test.cpp](tests/direct_dct_pls_test.cpp) |

本文描述当前实现，并明确记录了 AutoContrast 常量非零输入的 Python/CUDA 分支差异、Cutout 的纵向范围规则和 Translate 的正负取整差异；不把它们改写成其他库的理想化同名语义。

核对记录：在本机 `fastlanes-cuda` Python 环境中，使用 CPU 合成系数逐项对照了 14 个操作的 21 个强度／符号组合，分组实现与标量参考一致；另以独立构造的 IDCT 核对了水平翻转和正负 90° 旋转，并检查了本文数值例子、Python 示例语法和源码链接。上述检查没有执行原生 CUDA 端到端训练，也不覆盖已注明的全部边界输入。
