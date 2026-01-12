import os, json

base="ds_proxy_pred_patch"
out="cc-ds31t.json"

res = {}
has_patch=0

for file in os.listdir(base):
    with open(os.path.join(base, file)) as f:
        patch=f.read()
        if patch.strip():
            has_patch+=1
        res[file.strip(".diff")] = {"model_patch":patch}

with open(out, "w") as f:
    json.dump(res,f,indent=True)

print(has_patch, len(res))