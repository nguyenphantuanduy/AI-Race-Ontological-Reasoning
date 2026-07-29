# teacher_distillation.py

import os
import json
import random
import re
from glob import glob
from tqdm import tqdm

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)


# ==========================================================
# CONFIG
# ==========================================================

DATASET_DIR = "./dataset"

INPUT_DIR = os.path.join(
    DATASET_DIR,
    "input"
)

OUTPUT_DIR = os.path.join(
    DATASET_DIR,
    "output"
)


MODEL_NAME = "Qwen/Qwen3-8B"


NUM_GENERATE = 2000

BATCH_SIZE = 4

FEWSHOT_K = 3

FEWSHOT_MAX_TOKEN = 512

MAX_NEW_TOKENS = 1024


SEED = 42

random.seed(SEED)



# ==========================================================
# LABEL
# ==========================================================

NER_MAPPING = {

    "CHẨN_ĐOÁN": "DISEASE",

    "TRIỆU_CHỨNG": "SYMPTOM",

    "THUỐC": "DRUG",

    "KẾT_QUẢ_XÉT_NGHIỆM": "LAB_RESULT",

    "TÊN_XÉT_NGHIỆM": "TEST",
}


REVERSE_MAPPING = {
    v:k
    for k,v in NER_MAPPING.items()
}



# ==========================================================
# LOAD DATASET
# ==========================================================

def load_dataset():

    samples = []

    files = sorted(
        glob(
            os.path.join(
                INPUT_DIR,
                "*.txt"
            )
        ),
        key=lambda x:int(
            os.path.splitext(
                os.path.basename(x)
            )[0]
        )
    )


    for file in files:


        idx = os.path.splitext(
            os.path.basename(file)
        )[0]


        json_file = os.path.join(
            OUTPUT_DIR,
            f"{idx}.json"
        )


        if not os.path.exists(json_file):
            continue



        with open(
            file,
            "r",
            encoding="utf-8"
        ) as f:

            text = f.read()



        with open(
            json_file,
            "r",
            encoding="utf-8"
        ) as f:

            entities = json.load(f)



        samples.append(
            {
                "id": idx,
                "text": text,
                "entities": entities
            }
        )


    return samples



# ==========================================================
# RANDOM CROP
# ==========================================================

def random_crop_sample(
        sample,
        tokenizer,
        max_tokens=512
):

    text = sample["text"]


    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True
    )


    offsets = encoded["offset_mapping"]


    if len(offsets) <= max_tokens:
        return sample



    start_token = random.randint(
        0,
        len(offsets)-max_tokens
    )


    end_token = (
        start_token
        +
        max_tokens
        -
        1
    )


    char_start = offsets[start_token][0]

    char_end = offsets[end_token][1]


    crop_text = text[
        char_start:char_end
    ]


    crop_entities=[]


    for ent in sample["entities"]:


        s,e = ent["position"]


        if (
            s >= char_start
            and
            e <= char_end
        ):

            new_ent = ent.copy()

            new_ent["position"] = [
                s-char_start,
                e-char_start
            ]

            crop_entities.append(
                new_ent
            )


    return {
        "id": sample["id"],
        "text": crop_text,
        "entities": crop_entities
    }



# ==========================================================
# INSERT TAG
# ==========================================================

def insert_ner_tags(sample):

    text = sample["text"]


    entities = sorted(
        sample["entities"],
        key=lambda x:x["position"][0],
        reverse=True
    )


    for ent in entities:


        s,e = ent["position"]


        assert text[s:e] == ent["text"], (
            text[s:e],
            ent
        )


        label = NER_MAPPING.get(
            ent["type"],
            ent["type"]
        )


        text = (
            text[:s]
            +
            f"[[{label}]]"
            +
            text[s:e]
            +
            f"[[/{label}]]"
            +
            text[e:]
        )


    return text



# ==========================================================
# EXTRACT TAG
# ==========================================================

def extract_ner_tags(text):

    entities=[]

    clean=[]


    pattern = re.compile(
        r"\[\[(\w+)\]\](.*?)\[\[/\1\]\]",
        re.DOTALL
    )


    i=0

    clean_pos=0



    while i<len(text):


        m = pattern.search(
            text,
            i
        )


        if m is None:

            clean.append(
                text[i:]
            )

            break



        before=text[i:m.start()]


        clean.append(
            before
        )


        clean_pos += len(before)



        label=m.group(1)

        ent_text=m.group(2)



        start=clean_pos

        end=start+len(ent_text)



        entities.append(
            {
                "text":ent_text,

                "type":
                REVERSE_MAPPING.get(
                    label,
                    label
                ),

                "position":[
                    start,
                    end
                ]
            }
        )


        clean.append(
            ent_text
        )


        clean_pos=end

        i=m.end()



    return (
        "".join(clean),
        entities
    )



# ==========================================================
# FEWSHOT
# ==========================================================

def build_fewshot_examples(
        samples,
        tokenizer
):


    chosen=random.sample(
        samples[:100],
        FEWSHOT_K
    )


    examples=[]


    for i,s in enumerate(
        chosen,
        1
    ):


        crop=random_crop_sample(
            s,
            tokenizer,
            FEWSHOT_MAX_TOKEN
        )


        tagged=insert_ner_tags(
            crop
        )


        examples.append(
            f"""
### Example {i}

{tagged}
"""
        )


    return "\n\n".join(
        examples
    )



# ==========================================================
# PROMPT
# ==========================================================

def build_prompt(
        fewshot
):


    return f"""
Dưới đây là một số hồ sơ bệnh án tiếng Việt
đã được gắn nhãn thực thể.

Các nhãn:

- [[DISEASE]]...[[/DISEASE]]
- [[SYMPTOM]]...[[/SYMPTOM]]
- [[DRUG]]...[[/DRUG]]
- [[TEST]]...[[/TEST]]
- [[LAB_RESULT]]...[[/LAB_RESULT]]


{fewshot}


================================================


Hãy viết thêm MỘT hồ sơ bệnh án mới.

Yêu cầu:

- Văn bản hoàn toàn mới.
- Không sao chép ví dụ.
- Giữ phong cách hồ sơ bệnh án.
- Có thể gồm nhiều bệnh nhân.
- Chỉ sử dụng 5 loại nhãn trên.
- Chỉ trả về nội dung đã gắn nhãn.
- Không giải thích.
- Không có <think>.


Bắt đầu:
"""



# ==========================================================
# POSTPROCESS
# ==========================================================

def postprocess(text):

    text=re.sub(
        r"<think>.*?</think>",
        "",
        text,
        flags=re.S
    )


    text=re.sub(
        r"<\|.*?\|>",
        "",
        text
    )


    text=text.replace(
        "```",
        ""
    )


    return text.strip()



# ==========================================================
# MODEL
# ==========================================================

def load_model():


    tokenizer=AutoTokenizer.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True
    )


    config=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )


    model=AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=config,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )


    model.eval()


    return tokenizer,model



# ==========================================================
# GENERATE BATCH
# ==========================================================

@torch.no_grad()
def generate_batch(
        prompts,
        tokenizer,
        model
):


    inputs=tokenizer(
        prompts,
        padding=True,
        truncation=True,
        max_length=4096,
        return_tensors="pt"
    ).to(model.device)



    outputs=model.generate(
        **inputs,

        max_new_tokens=MAX_NEW_TOKENS,

        do_sample=True,

        temperature=0.9,

        top_p=0.95,

        top_k=50,

        repetition_penalty=1.1
    )



    results=[]


    for i,out in enumerate(outputs):


        prompt_len=inputs.input_ids[i].shape[0]


        text=tokenizer.decode(
            out[prompt_len:],
            skip_special_tokens=True
        )


        results.append(text)



    return results



# ==========================================================
# SAVE
# ==========================================================

def save_sample(
        idx,
        text,
        entities
):


    with open(
        os.path.join(
            INPUT_DIR,
            f"{idx}.txt"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        f.write(text)



    with open(
        os.path.join(
            OUTPUT_DIR,
            f"{idx}.json"
        ),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            entities,
            f,
            ensure_ascii=False,
            indent=2
        )



def get_next_id():

    ids=[]


    for f in glob(
        os.path.join(
            INPUT_DIR,
            "*.txt"
        )
    ):

        name=os.path.splitext(
            os.path.basename(f)
        )[0]


        if name.isdigit():
            ids.append(
                int(name)
            )


    return max(ids)+1 if ids else 1



# ==========================================================
# MAIN
# ==========================================================

def main():


    os.makedirs(
        INPUT_DIR,
        exist_ok=True
    )

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )



    samples=load_dataset()


    print(
        "Original:",
        len(samples)
    )


    tokenizer,model=load_model()



    current_id=get_next_id()


    count=0


    pbar=tqdm(
        total=NUM_GENERATE
    )


    while count < NUM_GENERATE:


        prompts=[]


        batch=min(
            BATCH_SIZE,
            NUM_GENERATE-count
        )


        for _ in range(batch):


            fewshot=build_fewshot_examples(
                samples,
                tokenizer
            )


            prompts.append(
                build_prompt(
                    fewshot
                )
            )



        outputs=generate_batch(
            prompts,
            tokenizer,
            model
        )



        for out in outputs:


            out=postprocess(
                out
            )


            text,entities=extract_ner_tags(
                out
            )


            if len(text)<100:
                continue


            if len(entities)==0:
                continue



            save_sample(
                current_id,
                text,
                entities
            )


            current_id+=1

            count+=1

            pbar.update(1)



            if count>=NUM_GENERATE:
                break



    pbar.close()


    print(
        "Generated:",
        count
    )



if __name__=="__main__":
    main()