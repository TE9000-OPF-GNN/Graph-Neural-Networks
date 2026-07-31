import re
content = open("Paper 3a draft - page 8.svg", encoding="utf-8", errors="replace").read()
texts = re.findall(r"<text[^>]*>(.*?)</text>", content, re.DOTALL)
for t in texts:
    clean = re.sub("<[^>]+>", "", t).strip()
    if clean:
        print(repr(clean))
