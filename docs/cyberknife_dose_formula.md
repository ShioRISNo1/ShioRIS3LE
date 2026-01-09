# CyberKnife線量計算式の確認

RayTracingモードにおける線量計算のソースコードを `src/cyberknife/dose_calculator.cpp` の
`CyberKnifeDoseCalculator::calculatePointDoseWithContext` で確認した。

該当箇所では以下のように線量が計算されている。

```
const double distanceTerm = (kReferenceSadMm * kReferenceSadMm) / (sad * sad);
const double dose = outputFactor * tmr * ocrRatio * distanceTerm * kRayTracingDoseScale;
```

ここで取得される係数の対応関係は次の通りである。

* `outputFactor` はコリメータサイズと深さに基づいて `BeamDataManager::getOutputFactor` から取得される出力係数 (OF)。【F:src/cyberknife/dose_calculator.cpp†L660-L675】
* `tmr` は `BeamDataManager::getTMRValue` により有効視野と深さから補間される Tissue Phantom Ratio (TPR)。【F:src/cyberknife/dose_calculator.cpp†L671-L681】
* `ocrRatio` は `BeamDataManager` あるいは OCR テーブルから補間した Off-Axis Ratio (OAR)。【F:src/cyberknife/dose_calculator.cpp†L663-L668】

`distanceTerm` は `(kReferenceSadMm * kReferenceSadMm) / (sad * sad)` で定義され、さらに `kRayTracingDoseScale` (現在は 0.01) によるセンチグレイ換算が行われるため、RayTracing モデルで得られる点線量は

\[
\text{Dose}_\text{RT} = 0.01 \times \mathrm{OF} \times \mathrm{TPR} \times \mathrm{OAR} \times \left(\frac{800}{\mathrm{SAD}}\right)^2
\]

となる。【F:src/cyberknife/dose_calculator.cpp†L1668-L1677】ここで `sad` はビーム中心軸長（単位はミリメートル）であり、距離項は \((800/\mathrm{SAD})^2\) に正規化され、0.01 倍で cGy に換算されている。【F:src/cyberknife/dose_calculator.cpp†L1668-L1677】

さらに、ボリューム線量合成では各ボクセルの点線量 `Dose_RT` にビーム重み `beamWeight` を直接掛け合わせて最終線量を得ており、ここでの `beamWeight` が MU に対応するように設計されている。【F:src/cyberknife/dose_calculator.cpp†L870-L879】従って、ShioRIS3 の実装は

\[
\text{Dose}_\text{voxel} = 0.01 \times \mathrm{beamWeight} \times \mathrm{OF} \times \mathrm{TPR} \times \mathrm{OAR} \times \left(\frac{800}{\mathrm{SAD}}\right)^2
\]

という形になっており、距離項も標準式 \(\mathrm{MU} \times (\mathrm{SAD}/800)^{-2}\) と一致する。

## DoseProfile 出力例（タブ区切り）

DoseProfile で出力されたサンプルデータをタブ区切りで掲載しておく。

```
Position (mm)	CT (HU)	Dose (Gy)
0.0	-19	0.11
1.0	-24	0.14
2.0	-34	0.19
3.0	-32	0.25
4.0	-30	0.38
5.0	-21	0.61
6.0	-27	1.11
```
