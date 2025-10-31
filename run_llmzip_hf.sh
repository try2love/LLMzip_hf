WIN_LEN=511
TEXT_FILE="test_texts/long_text.txt"
COMPRESSION_FOLDER="test_texts"
COMPRESSED_FILE_NAME="test_texts/long_text_511"
TARGET_FOLDER="meta-llama/Llama-2-7b-hf"
LORA_FOLDER=""

mkdir -p $COMPRESSION_FOLDER

# 运行命令
echo "开始运行 LLMzip..."
CUDA_VISIBLE_DEVICES=1 python LLMzip_run_hf.py \
  --ckpt_dir $TARGET_FOLDER \
  --tokenizer_path $TARGET_FOLDER/tokenizer.model \
  --win_len $WIN_LEN \
  --text_file $TEXT_FILE \
  --compression_folder $COMPRESSION_FOLDER \
  --compressed_file_name $COMPRESSED_FILE_NAME \
  --compression_alg ArithmeticCoding \
  --encode_decode 2 \
  --self_calculate_p True \
  --lora_dir $LORA_FOLDER

echo "完成！结果保存在: $COMPRESSION_FOLDER"s