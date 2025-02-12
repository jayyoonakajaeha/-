import argparse
import json
import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch.cuda.amp import autocast  # Mixed Precision을 위한 모듈 추가

from src.data import CustomDataset
from peft import LoraConfig, get_peft_model

# fmt: off
parser = argparse.ArgumentParser(prog="test", description="Testing about Conversational Context Inference.")

g = parser.add_argument_group("Common Parameter")
g.add_argument("--output", type=str, required=True, help="output filename")
g.add_argument("--model_id", type=str, required=True, help="huggingface model id")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--device", type=str, required=True, help="device to load the model")
# fmt: on


def main(args):

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16  # Mixed Precision을 위해 bfloat16 사용
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        device_map="auto",
        quantization_config=bnb_config
    )

    if args.tokenizer is None:
        args.tokenizer = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = 'right'  
    terminators = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("")]

    lora_config = LoraConfig(
        r=4,
        lora_alpha=8,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
        bias='none',
        task_type='CAUSAL_LM'
    )
    model = get_peft_model(model, lora_config)

    # 특정 레이어만 훈련 가능하도록 설정
    for name, param in model.named_parameters():
        if "lora" not in name:
            param.requires_grad = False

    model.eval()

    dataset = CustomDataset("resource/data/2_test.json", tokenizer)

    with open("resource/data/2_test.json", "r") as f:
        result = json.load(f)

    batch_size = 1  # 또는 GPU 메모리에 맞게 조정
    for idx in tqdm.tqdm(range(0, len(dataset), batch_size)):
        batch = dataset[idx:idx+batch_size]

        # 배치 차원 추가
        input_ids = batch[0].unsqueeze(0).to(0)  # CustomDataset에서 받은 토큰을 배치 차원으로 변환

        with autocast(enabled=True):
            outputs = model.generate(
                input_ids,
                max_new_tokens=512,
                eos_token_id=terminators,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
                num_beams=1,
            )
        
        # 결과 처리
        generated_tokens = outputs[:, input_ids.shape[-1]:]  # 입력 토큰 길이만큼을 슬라이스
        result[idx]["output"] = tokenizer.decode(generated_tokens.squeeze(), skip_special_tokens=True)

        # 메모리 해제
        del batch, input_ids, outputs
        torch.cuda.empty_cache()
        #print(result)
        
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=4))


if __name__ == "__main__":
    exit(main(parser.parse_args()))
