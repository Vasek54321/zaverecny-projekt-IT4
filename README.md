# Maturitní projekt IT4
## Silniční radar
 Radar po kalibraci měří rychlost auta pomocí dvou čar a následně vypočítá rychlost a zapíše ji do souboru. Pokud auto překročí povolenou rychlost, tak je zapsáno do souboru. 
## Cíle:
- Rozpoznat z videa auto
- Označit a následovat auto na videu
- Kalibrace kamery na co nejpřesnější měření
- Měřit rychlost aut živě a při překročení rychlosti vystřihnout auto a rozpoznat SPZ
##
### Konkurence
- https://github.com/dhananjaymenon/SpeedRadar-OpenCV-
- https://github.com/amplifiedengineering/opencv-radar
- https://github.com/Mega-Barrel/Speed-Detection-Using-OpenCV
### Způsob měření rychlosti: 
 Na změření rychlosti použiju kameru
### Použité technologie: 
 OpenCV
 EasyOCR
##
### Instalace
- Zkopírovat repositář
> git clone https://github.com/Vasek54321/zaverecny-projekt-IT4.git
- Vytvořit virtuální enviroment
> python -m venv .venv
> Linux: . .venv/bin/activate
> Windows: .venv\Scripts\activate
- Nainstalovat potřebné moduly
> pip install opencv-python easyocr 
- Spustit radar.py
##
