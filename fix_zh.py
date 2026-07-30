c = open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_zh.md','r',encoding='utf-8').read()

# Fix 1: Garbage \frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3} inserted before |w\|}
# Pattern: \frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3}|w\|}{...} -> \frac{\|w\|}{...}
c = c.replace(
    '\\frac{\\|w\\|^2 \\Delta_\\Sigma}{\\sigma_z^3}|w\\|}{\\sqrt{2\\pi}\\sigma_z}',
    '\\frac{\\|w\\|}{\\sqrt{2\\pi}\\sigma_z}'
)

# Fix 2: Corrupted sigmoid derivative expression
c = c.replace(
    '\\frac{\\|w\\|^2 \\Delta_\\Sigma}{\\sigma_z^3}partial \\ell}{\\partial z}',
    '\\frac{\\partial \\ell}{\\partial z}'
)

# Fix 3: Any remaining \frac{\|w\|^2 \Delta_\Sigma}{\sigma_z^3} before \partial
c = c.replace(
    '\\frac{\\|w\\|^2 \\Delta_\\Sigma}{\\sigma_z^3}partial',
    '\\partial'
)

# Fix 4: \frac{\frac{ -> \frac{ (nested)
while '\\frac{\\frac{' in c:
    c = c.replace('\\frac{\\frac{', '\\frac{')

# Fix 5: ||w|| with double backslashes
c = c.replace('\\|w\\\\\\|', '\\|w\\|')
c = c.replace('\\\\|w\\\\|', '\\|w\\|')

open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_zh.md','w',encoding='utf-8').write(c)
print('Done')
