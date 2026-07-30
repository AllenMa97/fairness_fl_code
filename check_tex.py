c = open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_en.tex','r',encoding='utf-8').read()
issues = []
for pat in ['|w\\\\|\\\\}\\\\{', 'Deltapartial', 'Deltalambda', '\\\\\\\\frac', "rac{\\\\|"]:
    count = c.count(pat)
    if count > 0:
        issues.append(f'{repr(pat)}: {count}')
for i, line in enumerate(c.split('\n')):
    if line.count('{') != line.count('}'):
        issues.append(f'Line {i+1}: unbalanced braces')
print(f'varsigma: {c.count("varsigma")}')
print(f'Issues: {len(issues)}')
for iss in issues[:15]: print(iss)
