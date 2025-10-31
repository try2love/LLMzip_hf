原始LLMzip仓库链接：[vcskaushik/LLMzip](https://github.com/vcskaushik/LLMzip)

项目是对文本数据使用LLM进行压缩，文本压缩的效果与文本的质量、LLM对文本的语义理解程度高度相关。本质上就是文本在LLM的PPL越低，压缩效果越好。

本仓库是在原始仓库的基础上添加了适配huggingface版本llama模型的功能，运行方案和原始仓库一致

额外添加了一些评估指标的计算，使得更加完善；额外添加了每一步下一个token预测的csv表和json汇总，直观展示下一个token的概率。

为什么要适配huggingface版本？因为[hiyouga/LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)对模型的微调是以hf版本为基础的，先把hf版本适配可以进一步尝试适配微调版本结果

待完成：

- [x] 适配微调后的hf版本模型
- [ ] 自适应win_len划分
- [ ] 中间结果以pkl文件形式存储
- [ ] 解码所需关键参数以文件名称的方式存储，而不是输出metrics.json文件

运行方案：

```bash
./run_llmzip_hf.sh
```

可以在sh文件中自定义一些必要的参数。

# 参数说明

## 模型与分词器配置

- **ckpt_dir**: `str`
  - 使用的LLaMA模型路径

- **tokenizer_path**: `str`
  - 分词器模型路径

- **lora_dir**: `str`
  - lora权重路径

## 文本处理参数

- **win_len**: `int`
  - 滑动窗口大小
  - 对于字数很少的文本，使用小窗口值（如4），否则可能导致反向压缩
  - 对于字数多的文本，推荐使用大窗口值（如511）
  - ⚠️ 上下文长度不能超过最大序列长度512

- **text_file**: `str`
  - 待压缩文本的文件路径

## 输出配置

- **compression_folder**: `str`
  - 压缩中间结果及最终结果的输出文件夹路径

- **compressed_file_name**: `str`
  - 压缩文件名称

## 序列处理参数

- **max_seq_len**: `int = 512`
  - 最大序列长度

- **max_batch_size**: `int = 32**
  - 最大批处理大小

## 压缩算法配置

- **compression_alg**: `str = 'ArithmeticCoding'`
  - 选择的压缩算法
  - 可选值：`ArithmeticCoding` / `RankZip` / `both`

- **encode_decode**: `int = 2`
  - 编码解码模式
  - `0`: 仅编码
  - `1`: 仅解码  
  - `2`: 编码和解码（默认）

## 高级选项

- **batched_encode**: `bool = False`
  - 是否使用批处理编码
  - ⚠️ 仅用于加速编码（理论熵计算），批处理编码不支持解码功能

- **with_context_start**: `bool = False`
  - 是否跳过初始上下文编码
  - 设置为`True`时，避免编码初始上下文，并在解码器处提供初始上下文

- **verify_save_decoded**: `int = 2`
  - 验证和保存解码结果选项
  - `0`: 不验证/不保存
  - `1`: 仅验证
  - `2`: 验证并保存（默认）

- **self_calculate_p**: `bool = False`
  - 是否计算评估指标
  - 包括压缩率和信息恢复率等评估指标的计算

## 文件命名说明

- 输入文件需包含扩展名
- 输出文件名不包含扩展名