# SpurerkennungWahlfach
Bei diesem Projekt handelt es sich um eine Prüfungsleistung für den Kurs Digitale Bildverarbeitung an der 
Dualen Hochschule Baden-Württemberg. 
Weitere Informationen zu den Anforderungen finden Sie unter [Aufgabe.pdf](aufgabe.pdf).

## Ordnerstruktur

```Text
|- CarSim_Build_v1: verschiedene Konfigurationen
    |  CarSim.exe: Starten CarSim
    
|- CarSim_Projekt_[...]: Verbindung und verarbeitung vom CarSim
    |  PythonServer_TCP_student.py: Start Lande detection für CarSim
    
|- config: verschiedene Konfigurationen

|- doc: Projekt-Dokumentation 
    |- class_description: Beschreibung des Quellcodes der erstellten Klassen
    |- vid: Aufnahme der verschiedenen Aufgaben
    
|- img: Bilder und Videos zum Projekt
    |- KITTI
    |- UDACITY
    
|- src: Feature Backbone
|  main.py: Start der 3 Videos
```

## Vorgehensweise

1. Größenänderung
2. "Schiebefenster" (Sliding Window)
   1. Segmentierung in Region of Interest
   2. Thresholding in HSV und Graustufe
   3. Histogramm erstellen
   4. Fenstern zeichen
      - Start beim Maximalwert des Histogramms
      - danach über den Mittelwert der Auswahl iterieren
   5. Polyfit mit ... - Filter
   6. Rücktransformation
3. Visualisierung


## Bonusaufgabe

challenge_video  
CarSim  

felix probiert:
* Objekte im Bild + (vlt Kennzeichen)
* Neuronalen Netzen Spurerkennung (bdd600k)

marc probiert:
* Kalmann/ ... - Filter

## Fazit

## Ausblick

## Quellen

