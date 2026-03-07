import io
import os
p = os.path.join(os.getcwd(), "major_project .ipynb")
print('Notebook path:', p)
with io.open(p, 'r', encoding='utf-8') as f:
    txt = f.read()
old = 'gemini-1.5-flash-latest'
new = 'gemini-2.5-flash'
if old in txt:
    txt2 = txt.replace(old, new)
    with io.open(p, 'w', encoding='utf-8') as f:
        f.write(txt2)
    print('Replaced occurrences of', old)
else:
    print('No occurrences found of', old)
