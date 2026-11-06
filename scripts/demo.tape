# vhs tape (https://github.com/charmbracelet/vhs): records docs/demo.gif.
# Regenerated in CI (.github/workflows/proof.yml). The corpus is the public
# synthetic transcript bundled with ctxfit (src/ctxfit/demo), never real data.
Output docs/demo.gif
Set FontSize 15
Set Width 1100
Set Height 520
Set Theme "Catppuccin Mocha"
Type "ctxfit replay src/ctxfit/demo"
Sleep 400ms
Enter
Sleep 4s
Type "ctxfit badge --out savings.json"
Enter
Sleep 2s
