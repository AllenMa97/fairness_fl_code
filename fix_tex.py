import re

with open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_en.tex', 'r', encoding='utf-8') as f:
    content = f.read()

# Fix 1: |w\|}{sqrt{2pi}sigma_z or sigma} -> \frac{\|w\|}{\sqrt{2\pi}...
# Covers both \sigma_z and \sigma variants in section 8
content = re.sub(
    r'\|w\\\|\}\{(\\sqrt\{2\\pi\}\\(?:sigma_z|sigma))\} \\cdot \\Delta',
    r'\\frac{\\|w\\\|}{\1} \\cdot \\Delta',
    content
)

# Fix 2: residual \Delta|w\|^2 in O() terms  
content = content.replace(
    '\\frac{\\|w\\|}{\\sqrt{2\\pi}\\sigma_z} \\cdot \\Delta|w\\|^2 \\Delta_{\\Sigma}}',
    '\\frac{\\|w\\|^2 \\Delta_{\\Sigma}}'
)

# Fix 3: missing \frac in various patterns
content = content.replace('\\partial \\ell}{\\partial z}', '\\frac{\\partial \\ell}{\\partial z}')
content = content.replace(
    '\\bar{g}_{g0,l} \\cdot \\bar{g}_{g1,l}}{\\|\\bar{g}_{g0,l}\\|',
    '\\frac{\\bar{g}_{g0,l} \\cdot \\bar{g}_{g1,l}}{\\|\\bar{g}_{g0,l}\\|'
)
content = content.replace('\\eta^2 (L + \\lambda M)}{2}', '\\frac{\\eta^2 (L + \\lambda M)}{2}')
content = content.replace("\\mu'_{g0}}{\\sigma'}", "\\frac{\\mu'_{g0}}{\\sigma'}")
content = content.replace("\\mu'_{g1}}{\\sigma'}", "\\frac{\\mu'_{g1}}{\\sigma'}")
content = content.replace(
    '\\mu_{g0,l} - \\mu_{g1,l}}{\\|\\mu_{g0,l}\\|',
    '\\frac{\\mu_{g0,l} - \\mu_{g1,l}}{\\|\\mu_{g0,l}\\|'
)

# Fix 4: \partial \mathcal{L}_{...}}{\partial -> \frac{\partial \mathcal{L}_{...}}{\partial
content = content.replace(
    '\\partial \\mathcal{L}_{\\mathrm{contrastive}}}{\\partial',
    '\\frac{\\partial \\mathcal{L}_{\\mathrm{contrastive}}}{\\partial'
)

# Fix 5: e^{\lambda^T Sigma lambda}{2} -> e^{\frac{\lambda^T Sigma lambda}{2}
content = content.replace(
    'e^{\\lambda^T \\Sigma_{g,l} \\lambda}{2}}',
    'e^{\\frac{\\lambda^T \\Sigma_{g,l} \\lambda}{2}}'
)

# Fix 6: \max(...)}{sigma'} -> \frac{\max(...)}{sigma'}
content = content.replace(
    '\\max(|\\mu\'_{g0}|, |\\mu\'_{g1}|)}{\\sigma\'}',
    '\\frac{\\max(|\\mu\'_{g0}|, |\\mu\'_{g1}|)}{\\sigma\'}'
)

# Fix 7: \alpha L_{eo}}{2\lambda_{eo}} -> \frac{\alpha L_{eo}}{2\lambda_{eo}}
content = content.replace(
    '\\alpha L_{\\mathrm{eo}}}{2\\lambda_{\\mathrm{eo}}}',
    '\\frac{\\alpha L_{\\mathrm{eo}}}{2\\lambda_{\\mathrm{eo}}}'
)

# Fix 8: Section 8 bounds: \sigma -> \sigma_z (where std dev, not sigmoid)
# In section 8 formulas (lines 470+), standalone \sigma should be \sigma_z
content = content.replace(
    '\\frac{\\|w\\|}{\\sqrt{2\\pi}\\sigma}',
    '\\frac{\\|w\\|}{\\sqrt{2\\pi}\\sigma_z}'
)
content = content.replace(
    '\\frac{\\|w\\|^2 \\Delta_{\\Sigma}}{\\sigma^3}',
    '\\frac{\\|w\\|^2 \\Delta_{\\Sigma}}{\\sigma_z^3}'
)
content = content.replace(
    '\\frac{\\|w\\|^2 \\Delta_\\Sigma}{\\sigma^2}',
    '\\frac{\\|w\\|^2 \\Delta_\\Sigma}{\\sigma_z^2}'
)
content = content.replace(
    '\\frac{1}{\\sigma\\sqrt{n}}',
    '\\frac{1}{\\sigma_z\\sqrt{n}}'
)

# Fix 9: rac{ -> \frac{ (corrupted \frac with missing \f)
while 'rac{\\|w\\|}' in content:
    content = content.replace('rac{\\|w\\|}', '\\frac{\\|w\\|}')

# Fix 10: \frac{\frac{ -> \frac{ 
while '\\frac{\\frac{' in content:
    content = content.replace('\\frac{\\frac{', '\\frac{')

with open(r'd:\最新PDFFed\fairness_fl_code\docs\proof_en.tex', 'w', encoding='utf-8') as f:
    f.write(content)

print('Done')
