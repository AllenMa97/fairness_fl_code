c = open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_zh.md','r',encoding='utf-8').read()
print(f"Total lines: {c.count(chr(10))+1}")
print(f"varsigma: {c.count('varsigma')}")

# Check corruption patterns
patterns = [
    ('partial \\ell}{\\partial z}', 'missing frac before partial'),
    ('|w\\|}{\\sqrt', '|w|}} without frac'),
    ('rac{\\|w\\|}', 'rac instead of frac'),
    ('\\\\frac', 'double backslash frac'),
    ('\\frac{\\frac', 'nested frac-frac'),
    ('Deltapartial', 'Deltapartial corruption'),
    ('Deltalambda', 'Deltalambda corruption'),
    ('Deltabar', 'Deltabar corruption'),
    ('Deltaeta', 'Deltaeta corruption'),
    ('Deltamax', 'Deltamax corruption'),
    ('Deltamu', 'Deltamu corruption'),
    ('Deltaalpha', 'Deltaalpha corruption'),
    ('\\|w\\\\\\|', 'double backslash in ||w||'),
]

for pat, desc in patterns:
    count = c.count(pat)
    if count > 0:
        # Find and print line numbers
        for i, line in enumerate(c.split('\n')):
            if pat in line:
                snippet = line.strip()[:120]
                print(f"  Line {i+1} [{desc}]: {snippet}")
        print(f"  [{desc}]: {count} occurrences total\n")
