import os
import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer

def test_converted_model(model_dir: str):
    print(f"正在从本地路径加载模型和分词器: {model_dir}")
    
    if not os.path.exists(model_dir):
        print(f"错误: 路径 {model_dir} 不存在，请检查转换输出目录是否正确。")
        sys.exit(1)

    # 1. 加载 Tokenizer
    # 转换脚本配置了自定义的 RwkvTokenizer，需要信任远程/本地代码
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        print("✅ 分词器 (Tokenizer) 加载成功！")
    except Exception as e:
        print(f"❌ 分词器加载失败，错误信息:\n{e}")
        sys.exit(1)

    # 2. 加载 Model
    # 自动读取 config.json 中的数据类型并加载权重
    try:
        print("正在加载模型权重（这可能需要一点时间）...")
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✅ 模型 (Model) 加载成功！运行设备: {model.device}")
    except Exception as e:
        print(f"❌ 模型加载失败，错误信息:\n{e}")
        sys.exit(1)

    # 3. 准备测试文本 (遵循 ChatML 格式)
    # prompt = "<|im_start|>user\n你好，请问你是谁？<|im_end|>\n<|im_start|>assistant\n"
    # inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    messages = [
        {"role": "user", "content": "你好，请问你是谁？"}
    ]

    # 自动应用你在转换时嵌入的 jinja 模板，并生成结构化文本
    prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True,
        enable_thinking=True  # 显式开启强制思考
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    # 4. 设置流式输出打印机
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=False)

    # 5. 测试生成
    print("\n--- 开始模型推理测试 ---")
    try:
        with torch.inference_mode():
            model.generate(
                **inputs,
                max_new_tokens=512,
                streamer=streamer,
                do_sample=True,
                temperature=1.0,
                top_p=0.5,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id
            )
        print("\n--- 推理测试结束 ---")
        print("🎉 恭喜！转换后的模型可以正常跑通推理。")
    except Exception as e:
        print(f"\n❌ 推理过程中发生错误:\n{e}")

if __name__ == "__main__":
    # 替换为你实际转换出来的 HF 模型文件夹路径
    # 例如刚刚那个未跟踪的模型如果转换出了新文件夹，把名字填在这里
    TARGET_HF_DIR = "/mnt/data/Models/rwkv/rwkv-step-4900-bf16-hf" 
    
    test_converted_model(TARGET_HF_DIR)