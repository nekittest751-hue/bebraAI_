# prepare_data.py - convert simple Q/A pairs in data.txt to train.jsonl
import json
out = open('train.jsonl','w',encoding='utf-8')
with open('data.txt','r',encoding='utf-8') as f:
    blocks = f.read().strip().split('\n\n')
    for b in blocks:
        lines = [l.strip() for l in b.split('\n') if l.strip()]
        if len(lines) >= 2:
            prompt = lines[0]
            completion = ' '.join(lines[1:])
            obj = {'prompt': prompt + '\n', 'completion': completion}
            out.write(json.dumps(obj, ensure_ascii=False) + '\n')
out.close()
print('train.jsonl written')
