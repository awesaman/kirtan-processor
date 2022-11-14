# kirtan-processor
To get started run
```
pip install -r requirements.txt
```
This will install the required modules.

Create an executable using
```
pyinstaller --onefile kirtan-processor.py
```
If you don't have pipinstaller, first run
```
pip install pyinstaller
```
or
```
pip3 install pyinstaller
```

<!-- To create DMG, use
```
brew install create-dmg
create-dmg \
  --volname "kirtan-processor" \
  --window-pos 200 120 \
  --window-size 600 300 \
  --hide-extension "kirtan-processor" \
  --app-drop-link 425 120 \
  "dist/KirtanProcessor.dmg" \
  "dist/dmg/"
``` -->